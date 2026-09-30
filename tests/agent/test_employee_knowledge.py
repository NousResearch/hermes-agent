import json
from types import SimpleNamespace

from agent.employee_prompt import prompt_parts
from agent.file_safety import get_write_denied_error
from agent.knowledge import guides_root
from tools.file_tools import read_file_tool
from tools.skill_provenance import set_current_write_origin, reset_current_write_origin


def test_guides_use_native_reads_and_only_review_writes_are_restricted(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    guide = guides_root()/'responsibility-authoring'/'guide.md'
    result = json.loads(read_file_tool(str(guide), offset=200, limit=1, task_id='employee-guides'))
    assert not result.get('error')
    assert len(result['content'].splitlines()) == 1
    assert '{profile_home}' not in result['content']
    assert get_write_denied_error(str(guide)) is None
    assert get_write_denied_error(str(tmp_path/'config.yaml')) is None
    token = set_current_write_origin('background_review')
    try:
        assert get_write_denied_error(str(tmp_path/'responsibilities'/'daily'/'schedules'/'job.yaml'))
        assert get_write_denied_error(str(tmp_path/'connections'/'mail'/'manual.md')) is None
    finally:
        reset_current_write_origin(token)
    parts = '\n'.join(prompt_parts(SimpleNamespace(valid_tool_names={'memory','delegate_task'})))
    assert str(tmp_path) in parts
    assert 'Service manuals:' in parts
    assert 'skill_manage' not in parts


def test_native_review_notices_include_confirmed_file_edits(tmp_path, monkeypatch):
    from tools import file_tools
    from tools.file_operations import ShellFileOperations
    from tools.environments.local import LocalEnvironment
    from agent.background_review import summarize_background_review_actions, _classify_review_result
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    monkeypatch.setattr(file_tools, '_get_file_ops', lambda *args: ops)
    path = str(tmp_path / 'connections/mail/manual.md')
    writes = [('write_file', {'path': path}, file_tools.write_file_tool(path, 'First procedure')),
              ('patch', {'path': path}, file_tools.patch_tool(path=path, old_string='First', new_string='Corrected')),
              ('patch', {'path': path}, file_tools.patch_tool(path=path, old_string='Absent', new_string='Invalid'))]
    messages = []
    for index, (tool, args, result) in enumerate(writes):
        messages.extend([{'role': 'assistant', 'tool_calls': [{'id': str(index), 'function': {'name': tool, 'arguments': json.dumps(args)}}]},
                         {'role': 'tool', 'tool_call_id': str(index), 'content': result}])
    actions = summarize_background_review_actions(messages, [])
    assert actions == ['Knowledge updated', 'Knowledge updated']
    assert _classify_review_result(actions) == 'knowledge'
    verbose = summarize_background_review_actions(messages, messages[:2], 'verbose')
    assert len(verbose) == 1 and path in verbose[0]
    assert summarize_background_review_actions(messages, [], 'off') == []


def test_prompt_remains_available_when_responsibility_root_needs_repair(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = tmp_path / 'responsibilities'
    root.write_text('accidentally a file')
    agent = SimpleNamespace(valid_tool_names={'memory', 'read_file'})
    prompt = '\n'.join(prompt_parts(agent))
    assert 'Responsibility index unavailable' in prompt
    assert 'No responsibilities yet' not in prompt
    assert 'Service manuals:' in prompt
    root.unlink()
    root.mkdir()
    repaired = '\n'.join(prompt_parts(agent))
    assert 'Responsibility index unavailable' not in repaired
    assert 'No responsibilities yet' in repaired


def test_native_guide_references_and_templates_resolve_through_file_tools(tmp_path, monkeypatch):
    import re

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = guides_root()
    hub = root / "employee"
    pending = [hub / "guide.md"]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        result = json.loads(read_file_tool(str(path), limit=1000, task_id="native-guide-links"))
        assert not result.get("error"), (path, result)
        content = result["content"]
        assert "{guides_root}" not in content
        if path.suffix != ".md":
            continue
        # References use the hub directory, including links between references.
        for match in re.finditer(r"`((?:references|templates)/[\w.-]+\.(?:md|js|mjs|yaml))`", content):
            pending.append((hub if path.is_relative_to(hub) else root / "responsibility-authoring") / match[1])
        for match in re.finditer(r"`((?:\.\./)+[\w./-]+\.md)`", content):
            candidate = (path.parent / match[1]).resolve()
            if not candidate.exists():
                candidate = (hub / match[1]).resolve()
            pending.append(candidate)
    assert hub / "references/native-mcp.md" in seen
    assert hub / "references/service-connections.md" in seen
    assert (root / "responsibility-authoring/guide.md").resolve() in seen
