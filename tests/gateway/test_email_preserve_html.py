"""Inbound formatting opt-in and plaintext compatibility (regression for #23695)."""

from email.mime.multipart import MIMEMultipart
from email.mime.message import MIMEMessage
from email.mime.text import MIMEText
from unittest.mock import AsyncMock

import pytest



HTML = '<p style="color: red; font-size: 20px">Rejected &amp; revised</p>'
PLAIN = 'Rejected & revised'


def _message(kind):
    if kind == 'single':
        msg = MIMEText(HTML, 'html', 'utf-8')
    elif kind == 'plain':
        msg = MIMEText(PLAIN, 'plain', 'utf-8')
    elif kind == 'empty':
        msg = MIMEText('', 'html', 'utf-8')
    else:
        msg = MIMEMultipart('mixed')
        attachment = MIMEText('<b>Not the body</b>', 'html', 'utf-8')
        attachment.add_header('Content-Disposition', 'attachment', filename='report.html')
        if kind == 'mixed_disposition':
            attachment.replace_header('Content-Disposition', 'Attachment; filename=report.html')
        elif kind == 'nested_attachment':
            attachment = MIMEMessage(MIMEText('<b>Not the body</b>', 'html', 'utf-8'))
            attachment.add_header('Content-Disposition', 'attachment', filename='forwarded.eml')
        msg.attach(attachment)
        alternatives = MIMEMultipart('alternative')
        alternatives.attach(MIMEText(PLAIN, 'plain', 'utf-8'))
        if kind == 'blank_before_html':
            alternatives.attach(MIMEText(' \n\t ', 'html', 'utf-8'))
        html = {'empty_html': '', 'blank_html': ' \n\t '}.get(kind, HTML)
        alternatives.attach(MIMEText(html, 'html', 'utf-8'))
        msg.attach(alternatives)
    msg['From'] = 'author@example.test'
    msg['Subject'] = 'Re: Review'
    msg['Message-ID'] = '<review@example.test>'
    if kind == 'unknown_charset':
        alternatives.get_payload()[1].set_param('charset', 'unknown-codec')
    return msg.as_bytes()


@pytest.mark.parametrize('flag', [None, False, 'false', True, 'true'])
@pytest.mark.parametrize('kind', ['single', 'multipart', 'plain', 'empty', 'empty_html', 'unknown_charset',
                                 'mixed_disposition', 'nested_attachment', 'blank_html', 'blank_before_html'])
def test_received_body_preserves_formatting_only_when_enabled(flag, kind):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter

    options = {'skip_attachments': True}
    if flag is not None:
        options['preserve_html'] = flag
    adapter = EmailAdapter(PlatformConfig.from_dict(options))
    parsed = adapter._parse_fetched_message(b'1', _message(kind))
    expected = '' if kind == 'empty' else PLAIN
    if flag in (True, 'true') and kind not in ('plain', 'empty', 'empty_html', 'blank_html'):
        expected = HTML
    assert parsed['body'] == expected
    assert parsed['attachments'] == []


@pytest.mark.asyncio
async def test_yaml_option_reaches_dispatched_email(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config, Platform
    from plugins.platforms.email.adapter import EmailAdapter

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('EMAIL_ALLOWED_USERS', 'author@example.test')
    config_path = tmp_path / 'config.yaml'
    for enabled in (True, False, True):
        config_path.write_text(
            'platforms:\n  email:\n    enabled: true\n'
            f'    preserve_html: {str(enabled).lower()}\n'
            '    skip_attachments: true\n'
            '    require_authenticated_sender: false\n', encoding='utf-8',
        )
        adapter = EmailAdapter(load_gateway_config().platforms[Platform.EMAIL])
        adapter.handle_message = AsyncMock()
        parsed = adapter._parse_fetched_message(b'1', _message('multipart'))
        await adapter._dispatch_message(parsed)
        event = adapter.handle_message.call_args.args[0]
        assert event.text == (HTML if enabled else PLAIN)
        assert event.source.user_id == 'author@example.test'
