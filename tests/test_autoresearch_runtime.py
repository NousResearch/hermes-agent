"""Runtime tests for Hermes AutoResearch."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from autoresearch import runtime, workspaces


def _write_param_project(root: Path) -> None:
    families_dir = root / '.hermes' / 'autoresearch' / 'families'
    families_dir.mkdir(parents=True)
    (families_dir.parent / 'project.yaml').write_text(
        dedent(
            '''
            project_id: demo-project
            description: Demo AutoResearch project
            default_cwd: .
            datasets:
              - demo.csv
            benchmarks:
              - baseline
            report_output_dir: research
            publish_target: telegram
            evaluator:
              evaluation: "eval-param|{candidate_json}|{result_json}"
              result_json: "{result_json}"
            '''
        ).strip()
        + '\n',
        encoding='utf-8',
    )
    (families_dir / 'params.yaml').write_text(
        dedent(
            '''
            family_id: params
            thesis: Search alpha values.
            commands:
              validation: "validate|{candidate_id}"
              evaluation: "eval-param|{candidate_json}|{result_json}"
              result_json: "{result_json}"
            mutation:
              mode: param_mutation
              population: 2
              survivors: 1
              parameter_space:
                alpha: [0.2, 0.8, 0.9]
            selection:
              primary_metric: metrics.holdout.score
              goal: maximize
              secondary_metrics:
                - metric: metrics.validation.score
                  min_delta: 0.0
            interesting_if:
              mode: all
              rules:
                - metric: champion.primary_delta
                  op: ">"
                  value: 0.1
            anchors:
              - candidate_id: baseline
                label: Baseline
                description: Baseline anchor
                parameters:
                  alpha: 0.2
            '''
        ).strip()
        + '\n',
        encoding='utf-8',
    )
    (root / 'demo.csv').write_text('alpha,score\n0.2,0.2\n', encoding='utf-8')


def _write_agent_patch_project(root: Path) -> None:
    families_dir = root / '.hermes' / 'autoresearch' / 'families'
    families_dir.mkdir(parents=True)
    (families_dir.parent / 'project.yaml').write_text(
        dedent(
            '''
            project_id: patch-project
            description: Agent patch AutoResearch project
            default_cwd: .
            report_output_dir: research
            evaluator:
              evaluation: "eval-patch|{workspace}|{result_json}"
              result_json: "{result_json}"
            '''
        ).strip()
        + '\n',
        encoding='utf-8',
    )
    (families_dir / 'patches.yaml').write_text(
        dedent(
            '''
            family_id: patches
            thesis: Improve the strategy file.
            commands:
              validation: "validate|{candidate_id}"
              evaluation: "eval-patch|{workspace}|{result_json}"
              result_json: "{result_json}"
            mutation:
              mode: agent_patch
              population: 1
              survivors: 1
              prompt: Raise the score without touching forbidden files.
            selection:
              primary_metric: metrics.holdout.score
              goal: maximize
            interesting_if:
              mode: all
              rules:
                - metric: champion.primary_delta
                  op: ">"
                  value: 0
            anchors:
              - candidate_id: baseline
                label: Baseline
                parameters: {}
            editable_files:
              - strategy.txt
            editable_markers:
              - file: strategy.txt
                start: "# START"
                end: "# END"
            '''
        ).strip()
        + '\n',
        encoding='utf-8',
    )
    (root / 'strategy.txt').write_text('# START\nalpha=1\n# END\n', encoding='utf-8')


def _fake_terminal_tool(command: str, **kwargs) -> str:
    del kwargs
    action, *parts = command.split('|')
    if action == 'validate':
        return json.dumps({'exit_code': 0, 'output': 'validated'})
    if action == 'eval-param':
        candidate_path = Path(parts[0])
        result_path = Path(parts[1])
        candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
        alpha = float(candidate['parameters'].get('alpha', 0.0))
        payload = {
            'metrics': {
                'holdout': {'score': alpha},
                'validation': {'score': alpha},
            }
        }
        result_path.write_text(json.dumps(payload), encoding='utf-8')
        return json.dumps({'exit_code': 0, 'output': 'ok'})
    if action == 'eval-patch':
        workspace = Path(parts[0])
        result_path = Path(parts[1])
        score = 0.0
        for line in (workspace / 'strategy.txt').read_text(encoding='utf-8').splitlines():
            if line.startswith('alpha='):
                score = float(line.split('=', 1)[1].strip())
                break
        payload = {
            'metrics': {
                'holdout': {'score': score},
                'validation': {'score': score},
            }
        }
        result_path.write_text(json.dumps(payload), encoding='utf-8')
        return json.dumps({'exit_code': 0, 'output': 'ok'})
    raise AssertionError(f'Unexpected command: {command}')



def _fake_run_command(command: str, workdir: Path, task_id: str | None) -> dict:
    del workdir, task_id
    return json.loads(_fake_terminal_tool(command))
def test_param_mutation_cycle_writes_report_and_summary(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    _write_param_project(root)
    monkeypatch.setattr(runtime, '_run_command', _fake_run_command)

    listed = runtime.list_projects(str(root))
    assert listed['count'] == 1
    assert listed['projects'][0]['project_id'] == 'demo-project'

    inspected = runtime.inspect_project(str(root))
    assert inspected['project']['project_id'] == 'demo-project'
    assert inspected['families'][0]['family_id'] == 'params'

    validation = runtime.validate_project(str(root))
    assert validation['valid'] is True

    result = runtime.research_cycle(
        project_root=str(root),
        family_id='params',
        population=2,
        survivors=1,
        seed=1,
    )

    assert result['status'] == 'completed'
    assert result['interesting']['verdict'] is True
    assert result['report_path']
    assert Path(result['report_path']).exists()
    assert result['selector']['champion']['primary_delta'] > 0
    assert 'champion' in result['summary']

    status_payload = runtime.status(run_id=result['run_id'], project_root=str(root))
    assert status_payload['status'] == 'completed'
    assert status_payload['report_path'] == result['report_path']

    runs = runtime.list_runs(project_root=str(root))
    assert runs['count'] == 1
    assert runs['runs'][0]['run_id'] == result['run_id']

    inspected_run = runtime.inspect_run(run_id=result['run_id'], project_root=str(root))
    assert 'AutoResearch Run' in inspected_run['report_preview']

    publish = runtime.publish_summary(run_id=result['run_id'], project_root=str(root))
    assert publish['summary'] == result['summary']
    assert publish['sent'] is False


def test_agent_patch_cycle_rejects_forbidden_files(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    _write_agent_patch_project(root)
    monkeypatch.setattr(runtime, '_run_command', _fake_run_command)

    def fake_run_agent_patch(*, workspace, **kwargs):
        del kwargs
        (workspace.path / 'forbidden.txt').write_text('nope\n', encoding='utf-8')
        return {'model': 'test-model', 'response': 'wrote forbidden file'}

    monkeypatch.setattr(runtime, '_run_agent_patch', fake_run_agent_patch)

    result = runtime.research_cycle(
        project_root=str(root),
        family_id='patches',
        population=1,
        survivors=1,
        seed=1,
        model='test-model',
    )

    assert result['status'] == 'completed'
    assert result['selector']['champion'] is None
    assert result['interesting']['verdict'] is False
    assert result['report_path'] is None
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['review']['accepted'] is False
    assert 'forbidden files' in result['candidates'][0]['review']['reasons'][0].lower()


def test_create_candidate_workspace_uses_git_worktree_when_repo_exists(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git is not installed')

    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'app.py').write_text('print("hello")\n', encoding='utf-8')

    subprocess.run(['git', 'init'], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'add', 'app.py'], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'commit', '-m', 'init'], cwd=root, check=True, capture_output=True, text=True)

    workspace = workspaces.create_candidate_workspace(root, 'run-1', 'candidate-1', ['app.py'])
    try:
        assert workspace.method == 'git_worktree'
        assert (workspace.path / 'app.py').exists()
        assert '.hermes/autoresearch/' in (root / '.gitignore').read_text(encoding='utf-8')

        (workspace.path / 'app.py').write_text('print("updated")\n', encoding='utf-8')
        assert workspaces.list_changed_files(workspace) == ['app.py']
    finally:
        subprocess.run(['git', 'worktree', 'remove', str(workspace.path), '--force'], cwd=root, check=True, capture_output=True, text=True)




