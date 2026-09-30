import json

from agent.employee_prompt import connection_guidance, file_keeping_guidance, responsibility_prompt
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
    parts = '\n'.join([connection_guidance(), responsibility_prompt()])
    assert str(tmp_path) in parts
    assert 'connections/guide.md' in parts
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
    prompt = '\n'.join([connection_guidance(), responsibility_prompt()])
    assert 'Responsibility index unavailable' in prompt
    assert 'No responsibilities yet' not in prompt
    assert 'connections/guide.md' in prompt
    root.unlink()
    root.mkdir()
    repaired = '\n'.join([connection_guidance(), responsibility_prompt()])
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
        # Resolve guide links as the model does after reading the parent guide.
        for match in re.finditer(r"`((?:(?:references|templates)/|(?:\.\./)+|(?:google-workspace|email|github)/)[\w./-]+\.(?:md|js|mjs|yaml))`", content):
            candidates = [(base / match[1]).resolve() for base in
                          (path.parent, path.parent.parent, hub)]
            target = next((candidate for candidate in candidates if candidate.is_file()), None)
            assert target is not None, (path, match[1])
            pending.append(target)
    assert hub / "references/native-mcp.md" in seen
    assert root / "connections/guide.md" in seen
    for service in ("google-workspace", "email", "github"):
        assert root / "connections" / service / "guide.md" in seen

    assert (root / "responsibility-authoring/guide.md").resolve() in seen
    assert (root / "file-keeping/guide.md").resolve() in seen


def test_filing_guidance_uses_active_profile_not_cwd(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    ambient = tmp_path / "ambient"
    profile = tmp_path / "selected"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    monkeypatch.chdir(tmp_path)
    token = set_hermes_home_override(profile)
    try:
        guidance = file_keeping_guidance()
        assert str(profile / "documents") in guidance
        assert str(profile / "repos") in guidance
        assert str(ambient) not in guidance
        assert str(guides_root() / "file-keeping" / "guide.md") in guidance
    finally:
        reset_hermes_home_override(token)
