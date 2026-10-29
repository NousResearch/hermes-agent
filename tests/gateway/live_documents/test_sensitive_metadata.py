"""Preserve credential filename policy across real event merging and gateway notes."""
import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def document_home(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / 'home'
    home.mkdir()
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(home)
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


def test_real_project_attachment_names_use_the_filename_policy(document_home):
    from plugins.platforms.discord.adapter_document_policy import _discord_attachment_name_requires_path_only

    root = Path(__file__).resolve().parents[3]
    env = root / '.env.example'
    readme = root / 'README.md'
    assert env.is_file() and readme.is_file()
    assert _discord_attachment_name_requires_path_only(env.name)
    assert not _discord_attachment_name_requires_path_only(readme.name)


def test_merged_real_attachment_paths_retain_path_only_gateway_notes(document_home):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import merge_pending_message_event
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.run_inbound import GatewayInboundMixin

    root = Path(__file__).resolve().parents[3]
    readme, env = root / 'README.md', root / '.env.example'
    body = readme.read_text(encoding='utf-8').splitlines()[0]
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'routes': {'documents': {'deliver': 'log'}}}))
    source = adapter._admission_source('documents', None, str(uuid4()))
    ordinary = MessageEvent(text=body, source=source, message_type=MessageType.DOCUMENT,
                            media_urls=[str(readme)], media_types=['text/markdown'], media_text_inlined=[False])
    private = MessageEvent(text='', source=source, message_type=MessageType.DOCUMENT,
                           media_urls=[str(env)], media_types=['text/plain'], media_text_inlined=[False],
                           metadata={'discord_path_only_attachment_paths': [str(env)]})
    queue = {}
    merge_pending_message_event(queue, source.chat_id, ordinary)
    merge_pending_message_event(queue, source.chat_id, private)
    merged = queue[source.chat_id]
    assert merged.media_urls == [str(readme), str(env)]
    assert merged.metadata['discord_path_only_attachment_paths'] == [str(env)]
    notes = GatewayInboundMixin._prepend_inbound_document_notes(merged, merged.text)
    assert 'may contain credentials or a private key' in notes
    assert env.name in notes and readme.name in notes and body in notes
    assert 'content has been included below' not in notes
