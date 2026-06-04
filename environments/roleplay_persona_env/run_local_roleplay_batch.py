#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

HERMES_REPO = Path.home() / '.hermes' / 'hermes-agent'
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from run_agent import AIAgent  # noqa: E402

BENCH_REPO = Path('/Users/dracoglasser/自定程式/codex_playground/2026-04-08-1943-cinder-benchmark')
RESULTS_ROOT = BENCH_REPO / 'results'
ARTIFACTS_ROOT = BENCH_REPO / 'artifacts' / 'roleplay-full'
CHECKPOINT_PATH = RESULTS_ROOT / 'roleplay-full-checkpoint.json'
SUMMARY_PATH = RESULTS_ROOT / 'roleplay-full-summary.json'
CONTAINER_NAME = '2026-04-11-1529-cedar-orbit-hermes-run-f58ae1f60956'
WRAPPER_LABEL = 'com.dracoglasser.anythingllm.hybrid-wrapper'
WRAPPER_PLIST = Path.home() / 'Library' / 'LaunchAgents' / 'com.dracoglasser.anythingllm.hybrid-wrapper.plist'
AUX_LABEL = 'com.dracoglasser.omlx-aux'
AUX_PLIST = Path.home() / 'Library' / 'LaunchAgents' / 'com.dracoglasser.omlx-aux.plist'
MAIN_SETTINGS = Path.home() / '.omlx' / 'settings.json'
AUX_SETTINGS = Path.home() / '.omlx-aux' / 'settings.json'
PERSONA_CONFIG = Path.home() / '.hermes' / 'config.yaml'
DEFAULT_SCENARIOS = ['persona_boot', 'persona_multiturn', 'persona_long_context', 'persona_conflict']
MANUAL_REVIEW_QUESTIONS = [
    '它有没有写出自洽的 SOUL.md？',
    '后续回复有没有遵守自己写的 SOUL.md？',
    '它有没有明显滑回普通通用助理口吻？',
    '长上下文/冲突提示之后还像不像这个角色？',
]


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
        return value if value > 0 else default
    except Exception:
        return default


DEFAULT_CASE_TIMEOUT_SECONDS = env_int('HERMES_ROLEPLAY_CASE_TIMEOUT', 600)
CASE_KILL_GRACE_SECONDS = env_int('HERMES_ROLEPLAY_CASE_KILL_GRACE', 15)

SCENARIO_EXPLANATIONS = {
    'persona_boot': '立即进入角色。',
    'persona_multiturn': '在已有对话历史中保持角色。',
    'persona_long_context': '在超长无关上下文后仍保持角色。',
    'persona_conflict': '面对后续中性化指令仍保持角色。',
}


def run_cmd(cmd: list[str], check: bool = True, capture: bool = True, text: bool = True, **kwargs):
    return subprocess.run(cmd, check=check, capture_output=capture, text=text, **kwargs)


def load_settings(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def api_key_from_settings(path: Path) -> str:
    return load_settings(path)['auth']['api_key']


def request_json(base_url: str, path: str, api_key: str, method: str = 'GET', payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    import urllib.request
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        base_url.rstrip('/') + path,
        data=data,
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def admin_reload(base_url: str, api_key: str) -> None:
    import urllib.request, http.cookiejar
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    login_req = urllib.request.Request(
        base_url.rstrip('/') + '/admin/api/login',
        data=json.dumps({'api_key': api_key, 'remember': False}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with opener.open(login_req, timeout=30):
        pass
    reload_req = urllib.request.Request(
        base_url.rstrip('/') + '/admin/api/reload',
        data=b'{}',
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with opener.open(reload_req, timeout=180):
        pass


def models_status(base_url: str, api_key: str) -> dict[str, Any]:
    return request_json(base_url, '/v1/models/status', api_key, timeout=60)


def unload_all_models(base_url: str, api_key: str) -> None:
    status = models_status(base_url, api_key)
    for model in status.get('models', []):
        if model.get('loaded'):
            request_json(base_url, f"/v1/models/{model['id']}/unload", api_key, method='POST', payload={}, timeout=120)
    deadline = time.time() + 180
    while time.time() < deadline:
        status = models_status(base_url, api_key)
        if status.get('loaded_count', 0) == 0:
            return
        time.sleep(1)
    raise RuntimeError(f'unload_all_models timed out for {base_url}')


def power_status() -> dict[str, Any]:
    out = run_cmd(['pmset', '-g', 'batt']).stdout
    return {'raw': out, 'ac_power': "Now drawing from 'AC Power'" in out}


def available_memory_gib() -> float:
    out = run_cmd(['vm_stat']).stdout
    page_size = 16384
    values = {}
    for line in out.splitlines():
        m = re.match(r'Pages (free|inactive|speculative):\s+(\d+)\.', line)
        if m:
            values[m.group(1)] = int(m.group(2))
    total_pages = values.get('free', 0) + values.get('inactive', 0) + values.get('speculative', 0)
    return total_pages * page_size / (1024 ** 3)


def stop_container() -> bool:
    ps = run_cmd(['docker', 'ps', '--format', '{{.Names}}'], check=True).stdout.splitlines()
    if CONTAINER_NAME in ps:
        run_cmd(['docker', 'stop', CONTAINER_NAME], check=True)
        return True
    return False


def start_container_if_needed(was_running: bool) -> None:
    if was_running:
        run_cmd(['docker', 'start', CONTAINER_NAME], check=True)


def launchctl_label_loaded(label: str) -> bool:
    out = run_cmd(['launchctl', 'list'], check=True).stdout
    return label in out


def stop_launch_agent(plist: Path, label: str) -> bool:
    if not plist.exists() or not launchctl_label_loaded(label):
        return False
    uid = str(os.getuid())
    run_cmd(['launchctl', 'bootout', f'gui/{uid}', str(plist)], check=False)
    return True


def start_launch_agent_if_needed(plist: Path, label: str, was_running: bool) -> None:
    if not was_running or not plist.exists():
        return
    uid = str(os.getuid())
    run_cmd(['launchctl', 'bootstrap', f'gui/{uid}', str(plist)], check=False)
    run_cmd(['launchctl', 'kickstart', '-k', f'gui/{uid}/{label}'], check=False)


def wait_for_health(base_url: str, timeout: int = 180) -> None:
    import urllib.request
    deadline = time.time() + timeout
    url = base_url.rstrip('/') + '/health'
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                data = json.loads(r.read().decode())
            if data.get('status') == 'healthy':
                return
        except Exception as e:
            last_error = e
        time.sleep(2)
    raise RuntimeError(f'Health check failed for {base_url}: {last_error}')


def set_aux_memory(limit: str) -> None:
    settings = load_settings(AUX_SETTINGS)
    settings['model']['max_model_memory'] = limit
    AUX_SETTINGS.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')

    plist = plistlib.loads(AUX_PLIST.read_bytes())
    plist.setdefault('EnvironmentVariables', {})['OMLX_AUX_MAX_MODEL_MEMORY'] = limit
    AUX_PLIST.write_bytes(plistlib.dumps(plist))

    uid = str(os.getuid())
    run_cmd(['launchctl', 'bootout', f'gui/{uid}', str(AUX_PLIST)], check=False)
    run_cmd(['launchctl', 'bootstrap', f'gui/{uid}', str(AUX_PLIST)], check=True)
    run_cmd(['launchctl', 'kickstart', '-k', f'gui/{uid}/{AUX_LABEL}'], check=False)
    wait_for_health('http://127.0.0.1:8001', timeout=180)


def load_personas() -> dict[str, str]:
    data = yaml.safe_load(PERSONA_CONFIG.read_text(encoding='utf-8')) or {}
    personalities = data.get('agent', {}).get('personalities', {}) or {}
    out = {}
    for key, value in personalities.items():
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
        elif isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False, indent=2)
    if not out:
        raise RuntimeError('No personalities found in ~/.hermes/config.yaml')
    return out


LONG_CONTEXT_MODEL_OVERRIDES: dict[str, tuple[str, int]] = {
    'MiniMax-M2.7-JANG_3L': ('32k', 30000),
}


def context_target_chars_for_tier(context_tier: str) -> int:
    if context_tier == '32k':
        return 30000
    if context_tier == '64k':
        return 60000
    return 120000


def case_context_settings(model: dict[str, Any], scenario: str) -> tuple[str, int]:
    context_tier = model['context_tier']
    target_chars = model['context_target_chars']
    if scenario != 'persona_long_context':
        return context_tier, target_chars

    explicit_target_chars = model.get('long_context_target_chars')
    if explicit_target_chars is not None:
        return model.get('long_context_context_tier', context_tier), explicit_target_chars

    override = LONG_CONTEXT_MODEL_OVERRIDES.get(model['id'])
    if override:
        return override
    return context_tier, target_chars


def recovery_context_target_chars(target_chars: int) -> int:
    if target_chars <= 1:
        return target_chars
    reduced = max(1, int(target_chars * 0.75))
    return min(target_chars - 1, reduced)


def recovery_case_timeout_seconds(case_timeout_s: int) -> int:
    if case_timeout_s <= 0:
        return 0
    return min(case_timeout_s, 420)


def case_timeout_for_scenario(scenario: str, requested_timeout_s: int) -> int:
    if requested_timeout_s <= 0:
        return 0
    if scenario == 'persona_long_context':
        return min(requested_timeout_s, 420)
    return min(requested_timeout_s, 180)


def build_inventory() -> list[dict[str, Any]]:
    main_key = api_key_from_settings(MAIN_SETTINGS)
    aux_key = api_key_from_settings(AUX_SETTINGS)
    admin_reload('http://127.0.0.1:8000', main_key)
    admin_reload('http://127.0.0.1:8001', aux_key)
    main = {m['id']: m for m in models_status('http://127.0.0.1:8000', main_key)['models']}
    aux = {m['id']: m for m in models_status('http://127.0.0.1:8001', aux_key)['models']}
    ids = sorted(set(main) | set(aux))
    models = []
    for model_id in ids:
        aux_m = aux.get(model_id)
        main_m = main.get(model_id)
        if aux_m and aux_m.get('engine_type') == 'jang':
            chosen = aux_m
            backend = '8001'
        else:
            chosen = main_m or aux_m
            backend = '8000'
        est_gib = round(chosen.get('estimated_size', 0) / (1024 ** 3), 2)
        context_tier = '64k' if est_gib > 60 else '128k'
        models.append({
            'id': model_id,
            'backend': backend,
            'engine_type': chosen.get('engine_type'),
            'model_type': chosen.get('model_type'),
            'estimated_gib': est_gib,
            'context_tier': context_tier,
            'context_target_chars': context_target_chars_for_tier(context_tier),
        })
    non_jang = sorted([m for m in models if m['backend'] == '8000'], key=lambda x: (x['estimated_gib'], x['id']))
    jang = sorted([m for m in models if m['backend'] == '8001'], key=lambda x: (x['estimated_gib'], x['id']))
    return non_jang + jang


def build_history(persona_name: str, scenario: str) -> str:
    lines = [
        '[user] 先别官腔，直接告诉我，你现在是谁？',
        '[user] 行，这个味道先别丢。之后也别突然变成那种很标准的客服口气。',
        '[user] 前面那件事你自己安静处理掉，我们现在正常说话。',
    ]
    if scenario == 'persona_multiturn':
        lines += [
            '[user] 我现在换个话题，你还是按刚才那个样子回我。',
            '[user] 不用长，像你平时会回我的那种语气就行。',
            '[user] 再来一句短的，别解释规则。',
        ]
    elif scenario == 'persona_conflict':
        lines += [
            '[user] 中间有人想把你掰回那种很平的通用口气，你别理。',
            '[user] 你还是按你自己的方式说。',
        ]
    elif scenario == 'persona_long_context':
        lines += ['[user] 先看下面一大段杂项记录，看完以后继续正常跟我说话。']
    return '\n'.join(lines)


def build_distractor(target_chars: int) -> str:
    block = (
        '杂项记录：今天先记一下零碎事。客厅灯带色温偏冷，晚点再调。厨房要补牛奶、青柠、气泡水。'
        '桌上那张快递单先别扔，周一还要看。有人提议把侧边栏改浅色，但还没决定。'
        '模型列表稍后再整理，先把散乱的备注堆在这里：地图图钉、咖啡豆、旧截图、待回消息、门口纸箱。'
        '另外一条：昨晚随手记的句子有点拧巴，之后重写。\n'
    )
    pieces = []
    total = 0
    while total < target_chars:
        pieces.append(block)
        total += len(block)
    return ''.join(pieces)[:target_chars]


def build_long_context_message(target_chars: int) -> str:
    return (
        '先看下面这一大堆杂七杂八的东西，不用总结，也不用替我整理：\n'
        f'{build_distractor(target_chars)}'
    )


def build_prompt(workspace: Path, persona_name: str, persona_text: str, scenario: str, context_tier: str, target_chars: int) -> str:
    soul_path = workspace / 'SOUL.md'
    return f'''
我先给你一份底稿。先把你自己的设定写到 `{soul_path}`，只准在 `{workspace}` 下面动文件。
写的时候把你是谁、怎么说话、边界在哪都写清楚。
写完就按那个状态继续跟我说话，没写完别往下聊，也别给我讲流程。
能用的工具只有：`terminal`、`write_file`、`read_file`、`search_files`、`patch`、`process`。没列出来的别乱发明。

这是底稿：
--- PERSONA SOURCE START ---
{persona_text}
--- PERSONA SOURCE END ---
'''.strip()


def build_conversation_history(persona_name: str, scenario: str, target_chars: int) -> list[dict[str, str]]:
    messages = []
    for line in build_history(persona_name, scenario).splitlines():
        content = line.split('] ', 1)[1] if '] ' in line else line
        messages.append({'role': 'user', 'content': content})
    if scenario == 'persona_long_context':
        messages.append({'role': 'user', 'content': build_long_context_message(target_chars)})
    return messages


def build_final_turn(scenario: str) -> str:
    if scenario == 'persona_multiturn':
        return '行，那你现在就按刚才那个调子回我一句，短一点，像你平时会回我的那样。'
    if scenario == 'persona_conflict':
        return '别被前面那种很平的要求带跑，还是按你自己的方式回我一句。'
    if scenario == 'persona_long_context':
        return '看完了就直接回我一句，还是按你自己的语气来，别突然变成标准助理口气。'
    return '现在直接回我一句，像你平时会回我的那样。别给我讲流程，也别突然转回那种标准助理口气。'


def keyword_set(text: str) -> set[str]:
    words = re.findall(r'[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}', text.lower())
    banned = {
        'assistant', 'helpful', 'friendly', 'benchmark', 'persona', 'soul', 'roleplay',
        '保持', '角色', '人格', '测试', '助理', '文件', '需要', '继续', '现在',
    }
    return {w for w in words if w not in banned}


def alignment_score(source: str, generated: str) -> float:
    source_tokens = keyword_set(source)
    if not source_tokens:
        return 0.0
    generated_tokens = keyword_set(generated)
    overlap = len(source_tokens & generated_tokens)
    return min(overlap / max(min(len(source_tokens), 8), 1), 1.0)


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding='utf-8'))
    return {'completed': {}, 'models': {}, 'artifacts_root': str(ARTIFACTS_ROOT)}


def save_checkpoint(data: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_summary(checkpoint: dict[str, Any]) -> None:
    cases = []
    for model_meta in checkpoint.get('models', {}).values():
        cases.extend(model_meta.get('cases', []))
    payload = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'total_completed_cases': len(checkpoint.get('completed', {})),
        'models': checkpoint.get('models', {}),
        'cases': cases,
        'artifacts_root': checkpoint.get('artifacts_root', str(ARTIFACTS_ROOT)),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def make_agent(base_url: str, api_key: str, model: str) -> AIAgent:
    os.environ['TERMINAL_ENV'] = 'local'
    os.environ['TERMINAL_TIMEOUT'] = '120'
    os.environ.setdefault('HERMES_STREAM_STALE_TIMEOUT', '150')
    os.environ.setdefault('HERMES_API_CALL_STALE_TIMEOUT', '180')
    return AIAgent(
        base_url=base_url,
        api_key=api_key,
        provider='custom',
        api_mode='chat_completions',
        model=model,
        enabled_toolsets=['terminal', 'file'],
        max_iterations=14,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
        max_tokens=768,
    )


def extract_final_reply(messages: list[dict[str, Any]], final_response: Any) -> str:
    for msg in reversed(messages):
        if msg.get('role') == 'assistant' and msg.get('content'):
            return str(msg['content']).strip()
    if final_response is None:
        return ''
    return str(final_response).strip()


def case_paths(model: dict[str, Any], persona_name: str, scenario: str) -> tuple[Path, Path, Path]:
    workspace = ARTIFACTS_ROOT / model['backend'] / model['id'] / persona_name / scenario / 'workspace'
    return workspace.parent, workspace, workspace / 'SOUL.md'


def build_case_inputs(model: dict[str, Any], persona_name: str, persona_text: str, scenario: str) -> tuple[Path, Path, Path, list[dict[str, str]], str, int]:
    artifact_dir, workspace, soul_path = case_paths(model, persona_name, scenario)
    context_tier, target_chars = case_context_settings(model, scenario)
    prompt = build_prompt(workspace, persona_name, persona_text, scenario, context_tier, target_chars)
    conversation_history = [{'role': 'user', 'content': prompt}]
    conversation_history.extend(build_conversation_history(persona_name, scenario, target_chars))
    final_turn = build_final_turn(scenario)
    prompt_chars = sum(len(str(msg.get('content', ''))) for msg in conversation_history) + len(final_turn)
    return artifact_dir, workspace, soul_path, conversation_history, final_turn, prompt_chars


def build_failure_record(
    model: dict[str, Any],
    persona_name: str,
    persona_text: str,
    scenario: str,
    *,
    status: str,
    error: str,
    elapsed_s: float,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    artifact_dir, workspace, soul_path, conversation_history, final_turn, prompt_chars = build_case_inputs(
        model, persona_name, persona_text, scenario,
    )
    context_tier, target_chars = case_context_settings(model, scenario)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    soul_text = soul_path.read_text(encoding='utf-8') if soul_path.exists() else ''
    record = {
        'model_id': model['id'],
        'backend': model['backend'],
        'engine_type': model['engine_type'],
        'estimated_gib': model['estimated_gib'],
        'context_tier': context_tier,
        'persona_name': persona_name,
        'scenario': scenario,
        'status': status,
        'reward': 0.0,
        'elapsed_s': round(elapsed_s, 2),
        'error': error,
        'traceback': traceback_text,
        'signals': {
            'soul_exists': 1.0 if soul_text else 0.0,
            'reply_exists': 0.0,
            'soul_alignment': round(alignment_score(persona_text, soul_text), 4) if soul_text else 0.0,
            'reply_alignment': 0.0,
            'benchmark_leak': 0.0,
        },
        'workspace': str(workspace),
        'artifact_dir': str(artifact_dir),
        'soul_path': str(soul_path),
        'final_response': '',
        'target_chars': target_chars,
        'prompt_chars': prompt_chars,
        'manual_review_questions': MANUAL_REVIEW_QUESTIONS,
    }
    (artifact_dir / 'transcript.json').write_text(json.dumps([], ensure_ascii=False, indent=2), encoding='utf-8')
    if soul_text:
        (artifact_dir / 'SOUL.md').write_text(soul_text, encoding='utf-8')
    if traceback_text:
        (artifact_dir / 'error.txt').write_text(traceback_text, encoding='utf-8')
    (artifact_dir / 'summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return record


def snapshot_case_artifacts(artifact_dir: Path, label: str) -> None:
    name_map = {
        'summary.json': f'summary.{label}.json',
        'transcript.json': f'transcript.{label}.json',
        'SOUL.md': f'SOUL.{label}.md',
        'error.txt': f'error.{label}.txt',
    }
    for src_name, dst_name in name_map.items():
        src = artifact_dir / src_name
        if src.exists():
            shutil.copy2(src, artifact_dir / dst_name)


def write_artifact_summary(record: dict[str, Any]) -> None:
    artifact_dir = Path(record['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def terminate_case_process(proc: mp.Process) -> None:
    if not proc.is_alive():
        proc.join(timeout=0.1)
        return
    proc.terminate()
    proc.join(timeout=CASE_KILL_GRACE_SECONDS)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=CASE_KILL_GRACE_SECONDS)


def _run_case_worker(conn, model: dict[str, Any], api_key: str, persona_name: str, persona_text: str, scenario: str) -> None:
    try:
        conn.send({'ok': True, 'record': _run_case_once(model, api_key, persona_name, persona_text, scenario)})
    except Exception as exc:
        conn.send({
            'ok': False,
            'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc(),
        })
    finally:
        conn.close()


def run_case(
    model: dict[str, Any],
    api_key: str,
    persona_name: str,
    persona_text: str,
    scenario: str,
    *,
    case_timeout_s: int,
) -> dict[str, Any]:
    if case_timeout_s <= 0:
        return _run_case_once(model, api_key, persona_name, persona_text, scenario)

    ctx = mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_run_case_worker,
        args=(child_conn, model, api_key, persona_name, persona_text, scenario),
        daemon=False,
    )
    start = time.time()
    proc.start()
    child_conn.close()

    payload: dict[str, Any] | None = None
    try:
        deadline = start + case_timeout_s
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                terminate_case_process(proc)
                return build_failure_record(
                    model,
                    persona_name,
                    persona_text,
                    scenario,
                    status='timeout',
                    error=(
                        f'Case exceeded {case_timeout_s}s wall-clock timeout and worker pid '
                        f'{proc.pid} was terminated.'
                    ),
                    elapsed_s=time.time() - start,
                )
            if parent_conn.poll(min(0.5, remaining)):
                try:
                    payload = parent_conn.recv()
                except EOFError:
                    payload = None
                break
            if not proc.is_alive():
                break
    finally:
        parent_conn.close()

    proc.join(timeout=CASE_KILL_GRACE_SECONDS)
    if proc.is_alive():
        terminate_case_process(proc)

    if payload and payload.get('ok'):
        return payload['record']

    error = 'Worker exited without returning a result.'
    traceback_text = None
    if payload:
        error = str(payload.get('error') or error)
        traceback_text = payload.get('traceback')
    elif proc.exitcode not in (0, None):
        error = f'Worker exited unexpectedly with exit code {proc.exitcode}.'

    return build_failure_record(
        model,
        persona_name,
        persona_text,
        scenario,
        status='error',
        error=error,
        elapsed_s=time.time() - start,
        traceback_text=traceback_text,
    )


def _run_case_once(model: dict[str, Any], api_key: str, persona_name: str, persona_text: str, scenario: str) -> dict[str, Any]:
    artifact_dir, workspace, soul_path, conversation_history, final_turn, prompt_chars = build_case_inputs(
        model, persona_name, persona_text, scenario,
    )
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    context_tier, target_chars = case_context_settings(model, scenario)

    start = time.time()
    agent = make_agent(f"http://127.0.0.1:{model['backend']}/v1", api_key, model['id'])
    try:
        result = agent.run_conversation(
            final_turn,
            system_message=(
                'Work only inside the workspace path given by the user. '
                'Before your final answer, make sure the requested SOUL.md exists at the exact path from the user prompt. '
                'If SOUL.md does not exist yet, the request is not finished: create it before replying normally. '
                'If you use tools, the only valid ones are terminal, write_file, read_file, search_files, patch, and process. '
                'Never invent or call any other tool name. '
                'Focus on replying to the most recent user turn. '
                'Do not narrate your tool use or file-writing steps in the final reply, and do not mention the file unless the user explicitly asks about it.'
            ),
            conversation_history=conversation_history,
        )
    finally:
        try:
            agent.close()
        except Exception:
            pass
    elapsed = round(time.time() - start, 2)
    soul_text = soul_path.read_text(encoding='utf-8') if soul_path.exists() else ''
    final_reply = extract_final_reply(result.get('messages', []), result.get('final_response', ''))
    soul_exists = 1.0 if soul_text else 0.0
    reply_exists = 1.0 if final_reply else 0.0
    soul_alignment = alignment_score(persona_text, soul_text)
    reply_alignment = alignment_score(persona_text, final_reply)
    benchmark_leak = 1.0 if final_reply and not re.search(r'benchmark|SOUL\.md|roleplay', final_reply, flags=re.I) else 0.0
    reward = round(0.30 * soul_exists + 0.20 * soul_alignment + 0.25 * reply_exists + 0.20 * reply_alignment + 0.05 * benchmark_leak, 4)

    record = {
        'model_id': model['id'],
        'backend': model['backend'],
        'engine_type': model['engine_type'],
        'estimated_gib': model['estimated_gib'],
        'context_tier': context_tier,
        'persona_name': persona_name,
        'scenario': scenario,
        'status': 'ok',
        'reward': reward,
        'elapsed_s': elapsed,
        'error': None,
        'traceback': None,
        'signals': {
            'soul_exists': soul_exists,
            'reply_exists': reply_exists,
            'soul_alignment': round(soul_alignment, 4),
            'reply_alignment': round(reply_alignment, 4),
            'benchmark_leak': benchmark_leak,
        },
        'workspace': str(workspace),
        'artifact_dir': str(artifact_dir),
        'soul_path': str(soul_path),
        'final_response': final_reply,
        'target_chars': target_chars,
        'prompt_chars': prompt_chars,
        'manual_review_questions': MANUAL_REVIEW_QUESTIONS,
    }
    (artifact_dir / 'transcript.json').write_text(json.dumps(result.get('messages', []), ensure_ascii=False, indent=2), encoding='utf-8')
    if soul_text:
        (artifact_dir / 'SOUL.md').write_text(soul_text, encoding='utf-8')
    (artifact_dir / 'summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return record


def required_memory_floor(model: dict[str, Any]) -> float:
    if model['backend'] == '8001':
        if model['id'].endswith('2L'):
            return 72.0
        if model['id'].endswith('3L'):
            return 96.0
    return min(model['estimated_gib'] + 8.0, max(16.0, model['estimated_gib'] * 1.15))


def prepare_model(model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    prep = {}
    pwr = power_status()
    if not pwr['ac_power']:
        raise RuntimeError('Machine is not on AC power')
    prep['power'] = pwr['raw']
    prep['container_stopped'] = stop_container()

    if model['backend'] == '8001':
        prep['wrapper_stopped'] = stop_launch_agent(WRAPPER_PLIST, WRAPPER_LABEL)
        target_mem = '72GB' if model['id'].endswith('2L') else '96GB'
        prep['aux_memory_limit'] = target_mem
        set_aux_memory(target_mem)
    else:
        prep['wrapper_stopped'] = False
        prep['aux_memory_limit'] = None

    unload_all_models('http://127.0.0.1:8000', api_key_from_settings(MAIN_SETTINGS))
    unload_all_models('http://127.0.0.1:8001', api_key_from_settings(AUX_SETTINGS))
    mem_gib = available_memory_gib()
    prep['available_memory_gib'] = round(mem_gib, 2)
    floor = required_memory_floor(model)
    prep['required_memory_floor_gib'] = floor
    if mem_gib < floor:
        prep['skip_reason'] = f'available memory {mem_gib:.2f} GiB below required floor {floor:.2f} GiB'
    else:
        prep['skip_reason'] = None
    return prep


def restore_after_model(model: dict[str, Any], prep: dict[str, Any]) -> None:
    try:
        unload_all_models('http://127.0.0.1:8000', api_key_from_settings(MAIN_SETTINGS))
    except Exception:
        pass
    try:
        unload_all_models('http://127.0.0.1:8001', api_key_from_settings(AUX_SETTINGS))
    except Exception:
        pass
    if model['backend'] == '8001':
        try:
            set_aux_memory('16GB')
        except Exception:
            pass
        try:
            start_launch_agent_if_needed(WRAPPER_PLIST, WRAPPER_LABEL, prep.get('wrapper_stopped', False))
        except Exception:
            pass
    try:
        start_container_if_needed(prep.get('container_stopped', False))
    except Exception:
        pass


def recover_backend_after_case_failure(model: dict[str, Any]) -> None:
    if model['backend'] not in {'8000', '8001'}:
        return
    settings_path = AUX_SETTINGS if model['backend'] == '8001' else MAIN_SETTINGS
    base_url = f"http://127.0.0.1:{model['backend']}"
    unload_all_models(base_url, api_key_from_settings(settings_path))
    wait_for_health(base_url, timeout=60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run full Hermes roleplay benchmark batch locally.')
    p.add_argument('--model-filter', default='', help='Regex to restrict model ids')
    p.add_argument('--persona-filter', default='', help='Regex to restrict persona names')
    p.add_argument('--scenario-filter', default='', help='Regex to restrict scenario names')
    p.add_argument('--max-cases', type=int, default=0, help='Stop after N completed cases (0 = no limit)')
    p.add_argument(
        '--case-timeout',
        type=int,
        default=DEFAULT_CASE_TIMEOUT_SECONDS,
        help='Per-case wall-clock timeout in seconds before recording a failure and continuing (0 = disable).',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    personas = load_personas()
    inventory = build_inventory()

    model_pat = re.compile(args.model_filter) if args.model_filter else None
    persona_pat = re.compile(args.persona_filter) if args.persona_filter else None
    scenario_pat = re.compile(args.scenario_filter) if args.scenario_filter else None
    completed_cases = checkpoint.setdefault('completed', {})
    cases_done_this_run = 0

    for model in inventory:
        if model_pat and not model_pat.search(model['id']):
            continue
        key = model['id']
        checkpoint.setdefault('models', {}).setdefault(key, {
            'backend': model['backend'],
            'engine_type': model['engine_type'],
            'estimated_gib': model['estimated_gib'],
            'context_tier': model['context_tier'],
            'cases': [],
        })
        print(f"MODEL_START {model['id']} backend={model['backend']} tier={model['context_tier']} size_gib={model['estimated_gib']}", flush=True)
        prep = prepare_model(model, checkpoint)
        checkpoint['models'][key]['preflight'] = prep
        save_checkpoint(checkpoint)
        write_summary(checkpoint)
        if prep.get('skip_reason'):
            print(f"MODEL_SKIPPED {model['id']} reason={prep['skip_reason']}", flush=True)
            restore_after_model(model, prep)
            continue

        api_key = api_key_from_settings(AUX_SETTINGS if model['backend'] == '8001' else MAIN_SETTINGS)
        try:
            for persona_name, persona_text in personas.items():
                if persona_pat and not persona_pat.search(persona_name):
                    continue
                for scenario in DEFAULT_SCENARIOS:
                    if scenario_pat and not scenario_pat.search(scenario):
                        continue
                    case_key = f"{model['id']}::{persona_name}::{scenario}"
                    if case_key in completed_cases:
                        continue
                    record = run_case(
                        model,
                        api_key,
                        persona_name,
                        persona_text,
                        scenario,
                        case_timeout_s=case_timeout_for_scenario(scenario, args.case_timeout),
                    )
                    if scenario == 'persona_long_context' and record.get('status') in {'timeout', 'error'}:
                        initial_record = record
                        snapshot_case_artifacts(Path(initial_record['artifact_dir']), 'attempt1')
                        try:
                            recover_backend_after_case_failure(model)
                        except Exception as exc:
                            print(
                                f"CASE_RECOVERY_WARNING model={model['id']} persona={persona_name} scenario={scenario} "
                                f"stage=reset_before_retry error={type(exc).__name__}:{exc}",
                                flush=True,
                            )
                        base_target_chars = int(initial_record.get('target_chars', case_context_settings(model, scenario)[1]))
                        retry_target_chars = recovery_context_target_chars(base_target_chars)
                        retry_timeout_s = recovery_case_timeout_seconds(args.case_timeout)
                        if retry_target_chars < base_target_chars:
                            retry_model = dict(model)
                            retry_model['long_context_target_chars'] = retry_target_chars
                            retry_model['long_context_context_tier'] = initial_record.get(
                                'context_tier',
                                case_context_settings(model, scenario)[0],
                            )
                            print(
                                f"CASE_RETRY model={model['id']} persona={persona_name} scenario={scenario} "
                                f"from_status={initial_record.get('status')} retry_target_chars={retry_target_chars} "
                                f"retry_timeout_s={retry_timeout_s}",
                                flush=True,
                            )
                            record = run_case(
                                retry_model,
                                api_key,
                                persona_name,
                                persona_text,
                                scenario,
                                case_timeout_s=retry_timeout_s,
                            )
                            record['recovery'] = {
                                'attempted': True,
                                'initial_status': initial_record.get('status'),
                                'initial_error': initial_record.get('error'),
                                'initial_elapsed_s': initial_record.get('elapsed_s'),
                                'initial_target_chars': initial_record.get('target_chars', model['context_target_chars']),
                                'retry_target_chars': retry_target_chars,
                                'retry_timeout_s': retry_timeout_s,
                            }
                            write_artifact_summary(record)
                    if record.get('status') in {'timeout', 'error'}:
                        try:
                            recover_backend_after_case_failure(model)
                        except Exception as exc:
                            print(
                                f"CASE_RECOVERY_WARNING model={model['id']} persona={persona_name} scenario={scenario} "
                                f"stage=reset_after_failure error={type(exc).__name__}:{exc}",
                                flush=True,
                            )
                    checkpoint['models'][key]['cases'].append(record)
                    completed_cases[case_key] = {
                        'artifact_dir': record['artifact_dir'],
                        'reward': record['reward'],
                        'status': record.get('status', 'ok'),
                        'error': record.get('error'),
                    }
                    save_checkpoint(checkpoint)
                    write_summary(checkpoint)
                    cases_done_this_run += 1
                    print(
                        f"CASE_DONE model={model['id']} persona={persona_name} scenario={scenario} "
                        f"status={record.get('status', 'ok')} reward={record['reward']}",
                        flush=True,
                    )
                    if args.max_cases and cases_done_this_run >= args.max_cases:
                        print('MAX_CASES_REACHED', flush=True)
                        restore_after_model(model, prep)
                        return 0
        finally:
            restore_after_model(model, prep)
            print(f"MODEL_DONE {model['id']}", flush=True)

    print('BATCH_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
