from hermes_cli.session_transcript import project_transcript


def options(**changes):
    result = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                  include_files=True, include_links=True, include_asides=True, collapse_details=True)
    result.update(changes)
    return result


def test_stored_speaker_timestamp_and_reasoning_are_projected_without_private_fields():
    data = {'id': 's', 'started_at': 1700000000, 'model': 'm', 'system_prompt': 'PRIVATE',
            'messages': [{'id': 7, 'role': 'assistant', 'name': 'Ada', 'timestamp': 1700000001,
                          'content': 'answer', 'reasoning': 'thought', 'reasoning_details': 'opaque'}]}
    doc = project_transcript(data, options(include_reasoning=True))
    turn = doc['messages'][0]
    assert turn['speaker'] == 'Ada' and turn['id'] == '7'
    assert turn['timestamp'] == '2023-11-14T22:13:21+00:00'
    assert turn['reasoning'] == 'thought'
    assert doc['created_at'] == '2023-11-14T22:13:20+00:00' and doc['model'] == 'm'
    assert 'PRIVATE' not in str(doc) and 'opaque' not in str(doc)


def test_multimodal_parts_filter_images_without_leaking_data():
    data = {'id': 's', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'hello'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aGVsbG8='}},
        {'type': 'video_url', 'video_url': {'url': 'https://example.com/v.mp4'}},
    ]}]}
    turn = project_transcript(data, options(include_images=False))['messages'][0]
    assert turn['text'] == 'hello'
    assert [a['kind'] for a in turn['attachments']] == ['video']
    assert 'aGVsbG8=' not in str(turn)


def test_valid_raster_data_url_is_embedded_and_invalid_payload_refused():
    import base64
    payload = base64.b64encode(b'\x89PNG\r\n\x1a\nsynthetic').decode()
    data = {'id': 's', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + payload}},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,!!!'}},
    ]}]}
    images = project_transcript(data, options())['messages'][0]['attachments']
    assert images[0]['data_url'] == 'data:image/png;base64,' + payload
    assert 'data_url' not in images[1] and images[1]['unavailable']


def test_tool_calls_are_only_present_when_requested():
    data = {'id': 's', 'messages': [{'role': 'assistant', 'content': 'checking', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'shell', 'arguments': '{"cmd":"pwd"}'}}]}]}
    without = project_transcript(data, options())['messages'][0]
    with_tools = project_transcript(data, options(include_tools=True))['messages'][0]
    assert 'tool_calls' not in without
    assert with_tools['tool_calls'] == [{'name': 'shell', 'arguments': '{"cmd":"pwd"}'}]


def test_tool_identity_is_preserved_per_occurrence_without_inventing_links():
    data = {'id': 's', 'messages': [
        {'role': 'assistant', 'content': 'checking', '_tool_call_uids': {'repeat': ['uid-one', 'uid-two']},
         'tool_calls': [{'id': 'repeat', 'function': {'name': 'shell', 'arguments': '{}'}},
                        {'id': 'repeat', 'function': {'name': 'shell', 'arguments': '{}'}},
                        {'function': {'name': 'shell', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'second', 'tool_call_id': 'repeat', '_tool_call_uid': 'uid-two'},
        {'role': 'tool', 'content': 'unlinked'},
    ]}
    turns = project_transcript(data, options(include_tools=True))['messages']
    assert turns[0]['tool_calls'] == [
        {'name': 'shell', 'arguments': '{}', 'id': 'repeat', 'uid': 'uid-one'},
        {'name': 'shell', 'arguments': '{}', 'id': 'repeat', 'uid': 'uid-two'},
        {'name': 'shell', 'arguments': '{}'},
    ]
    assert turns[1]['tool_call_id'] == 'repeat' and turns[1]['tool_call_uid'] == 'uid-two'
    assert 'tool_call_id' not in turns[2] and 'tool_call_uid' not in turns[2]
    without = project_transcript(data, options())['messages']
    assert len(without) == 1 and 'tool_calls' not in without[0]


def test_redaction_never_turns_distinct_sensitive_tool_ids_into_a_false_match():
    from hermes_cli.session_transcript import redact_transcript
    data = {'id': 's', 'messages': [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'sk-abcdefghijklmnopqrstuvwxyz1234567890', 'function': {'name': 'shell', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'result', 'tool_call_id': 'sk-zyxwvutsrqponmlkjihgfedcba1234567890'}]}
    doc = redact_transcript(project_transcript(data, options(include_tools=True)))
    assert 'id' not in doc['messages'][0]['tool_calls'][0]
    assert 'tool_call_id' not in doc['messages'][1]
    assert 'sk-' not in str(doc)


def test_real_database_lineage_preserves_stored_tool_occurrences(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / 'identity.db')
    try:
        db.create_session('parent', source='webui')
        db.append_messages_batch('parent', [
            {'role': 'assistant', 'content': 'working', 'reasoning': 'thinking', 'tool_calls': [
                {'id': 'a', 'function': {'name': 'shell', 'arguments': '{}'}},
                {'id': 'b', 'function': {'name': 'shell', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'b', 'content': 'second'},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'first'},
        ])
        db.end_session('parent', 'compression')
        db.create_session('child', source='webui', parent_session_id='parent')
        db.append_message('child', 'assistant', 'answer')
        rows = db.get_messages('child', include_compacted=True, include_ancestors=True)
        doc = project_transcript({'id': 'child', 'messages': rows}, options(include_tools=True, include_reasoning=True), home=tmp_path)
        turns = doc['messages']
        calls = turns[0]['tool_calls']
        assert turns[0]['reasoning'] == 'thinking'
        assert [call['id'] for call in calls] == ['a', 'b']
        assert [turn['tool_call_id'] for turn in turns[1:3]] == ['b', 'a']
        assert turns[1]['tool_call_uid'] == calls[1]['uid']
        assert turns[2]['tool_call_uid'] == calls[0]['uid']
        assert turns[3]['text'] == 'answer'
    finally:
        db.close()


def test_only_managed_cache_images_are_read(tmp_path, monkeypatch):
    import base64
    home = tmp_path / 'home'
    root = home / 'cache' / 'images'
    root.mkdir(parents=True)
    image = b'\x89PNG\r\n\x1a\nmanaged'
    (root / 'good.png').write_bytes(image)
    (tmp_path / 'secret.png').write_bytes(image)
    monkeypatch.setenv('HERMES_HOME', str(home))
    data = {'id': 's', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': str(root / 'good.png')}},
        {'type': 'image_url', 'image_url': {'url': str(tmp_path / 'secret.png')}},
    ]}]}
    attachments = project_transcript(data, options())['messages'][0]['attachments']
    assert attachments[0]['data_url'] == 'data:image/png;base64,' + base64.b64encode(image).decode()
    assert 'data_url' not in attachments[1] and attachments[1]['unavailable']


def test_cache_read_rechecks_opened_handle_against_swapped_target(tmp_path, monkeypatch):
    from pathlib import Path
    home = tmp_path / 'home'
    root = home / 'cache' / 'images'
    root.mkdir(parents=True)
    target = root / 'image.png'
    target.write_bytes(b'\x89PNG\r\n\x1a\nintended')
    secret = tmp_path / 'secret.png'
    secret.write_bytes(b'\x89PNG\r\n\x1a\nsecret')
    monkeypatch.setenv('HERMES_HOME', str(home))
    original = Path.open
    def swap(self, *args, **kwargs):
        return original(secret if self == target else self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', swap)
    data = {'id': 's', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': str(target)}}]}]}
    attachment = project_transcript(data, options())['messages'][0]['attachments'][0]
    assert 'data_url' not in attachment


def test_stored_image_hint_is_excluded_when_images_are_off():
    data = {'id': 's', 'messages': [{'role': 'user', 'content': 'Look\n[Image attached at: /private/image.png]'}]}
    turn = project_transcript(data, options(include_images=False))['messages'][0]
    assert turn['text'] == 'Look'
    assert '/private/image.png' not in str(turn)


def test_unc_image_reference_never_resolves_network_path(tmp_path, monkeypatch):
    from pathlib import Path
    home = tmp_path / 'home'
    (home / 'cache' / 'images').mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    original = Path.resolve
    def guarded(self, *args, **kwargs):
        assert not str(self).startswith('\\\\'), 'UNC path must never be resolved'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', guarded)
    original_symlink = Path.is_symlink
    def guarded_symlink(self):
        assert not str(self).startswith('\\\\'), 'UNC path must never be probed'
        return original_symlink(self)
    monkeypatch.setattr(Path, 'is_symlink', guarded_symlink)
    data = {'id': 's', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': r'\\server\share\image.png'}}]}]}
    attachment = project_transcript(data, options())['messages'][0]['attachments'][0]
    assert 'data_url' not in attachment


def test_transient_btw_is_disclosed_without_inventing_turns():
    doc = project_transcript({'id': 's', 'messages': [{'role': 'user', 'content': 'hello'}]}, options())
    assert any('/btw' in warning and 'not' in warning.lower() for warning in doc['warnings'])
    assert [turn['text'] for turn in doc['messages']] == ['hello']


def test_developer_instruction_is_not_a_readable_turn():
    data = {'id': 's', 'messages': [{'role': 'developer', 'content': 'private instruction'},
                                     {'role': 'user', 'content': 'visible'}]}
    assert [m['text'] for m in project_transcript(data, options())['messages']] == ['visible']


def test_text_image_hint_never_triggers_a_cache_read(tmp_path, monkeypatch):
    from pathlib import Path
    home = tmp_path / 'home'
    root = home / 'cache' / 'images'
    root.mkdir(parents=True)
    image = root / 'image.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\nmanaged')
    monkeypatch.setenv('HERMES_HOME', str(home))
    def forbidden(*args, **kwargs):
        raise AssertionError('untrusted text triggered a filesystem read')
    monkeypatch.setattr(Path, 'open', forbidden)
    data = {'id': 's', 'messages': [{'role': 'user', 'content': f'[Image attached at: {image}]'}]}
    attachment = project_transcript(data, options())['messages'][0]['attachments'][0]
    assert attachment['reference'] == str(image) and attachment['unavailable']



def test_quoted_media_directive_with_spaces_is_filtered():
    row = {'id': 's', 'messages': [{'role': 'assistant', 'content': 'MEDIA:"/tmp/my video.mp4"'}]}
    turn = project_transcript(row, options())['messages'][0]
    assert turn['attachments'][0]['reference'] == '/tmp/my video.mp4'
    assert turn['attachments'][0]['kind'] == 'video'
    assert project_transcript(row, options(include_videos=False))['messages'][0]['text'] == ''


def test_first_line_ordinary_at_ref_in_prose_is_not_attachment():
    text = '@file:report.pdf explains the result'
    turn = project_transcript({'id': 's', 'messages': [{'role': 'user', 'content': text}]}, options(include_files=False))['messages'][0]
    assert turn['text'] == text


def test_structured_text_parts_parse_inline_media_directives():
    row = {'id': 's', 'messages': [{'role': 'assistant', 'content': [
        {'type': 'text', 'text': 'Here is MEDIA:/tmp/movie.mp4 and more'},
        {'type': 'input_file', 'file': {'filename': 'report.pdf', 'file_data': 'opaque'}}]}]}
    enabled = project_transcript(row, options())['messages'][0]
    assert [a['kind'] for a in enabled['attachments']] == ['video', 'file']
    disabled = project_transcript(row, options(include_videos=False, include_files=False))['messages'][0]
    assert 'MEDIA:' not in disabled['text'] and 'more' in disabled['text']


def test_canonical_media_directives_filter_without_removing_prose_refs():
    text = '@file:report.pdf\n@folder:`folder name`\n\nSee @file:notes in prose.\nMEDIA:/tmp/clip.mp4\n[Image attached at: /tmp/a.png]'
    row = {"id": "s", "messages": [{"role": "user", "content": text}]}
    enabled = project_transcript(row, options())["messages"][0]
    assert [a["kind"] for a in enabled["attachments"]] == ["file", "file", "video", "image"]
    disabled = project_transcript(row, options(include_files=False, include_videos=False, include_images=False))["messages"][0]
    assert disabled["text"].strip() == "See @file:notes in prose."


def test_leading_image_reference_is_filtered_when_images_disabled():
    data = {'id': 's', 'messages': [{'role': 'user', 'content': '@image:/tmp/secret.png\n\nDescribe this'}]}
    turn = project_transcript(data, options(include_images=False))['messages'][0]
    assert 'secret.png' not in str(turn)
    assert 'Describe this' in turn['text']


def test_leading_image_and_video_file_refs_obey_their_media_options():
    text = '@file:/tmp/picture.png\n@file:/tmp/movie.mp4\n@file:/tmp/report.pdf\n\nhello'
    data = {'id': 's', 'messages': [{'role': 'user', 'content': text}]}
    turn = project_transcript(data, options(include_images=False, include_videos=False))['messages'][0]
    assert [a['kind'] for a in turn['attachments']] == ['file']
    assert 'picture.png' not in str(turn) and 'movie.mp4' not in str(turn)


def test_profile_and_explicit_workspace_image_roots(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace" / ".hermes" / "attachments"
    roots = [home / "images", home / "image_cache", home / "attachments", workspace]
    for root in roots:
        root.mkdir(parents=True)
        (root / "a.png").write_bytes(bytes.fromhex('89504e470d0a1a0a') + b'image')
    content = [{"type": "image_url", "image_url": {"url": str(root / "a.png")}} for root in roots]
    data = {"id": "s", "messages": [{"role": "user", "content": content}]}
    attachments = project_transcript(data, options(), home=home, trusted_roots=[workspace])["messages"][0]["attachments"]
    assert all("data_url" in a for a in attachments)
    untrusted = project_transcript(data, options(), home=home)["messages"][0]["attachments"]
    assert "data_url" not in untrusted[-1]
