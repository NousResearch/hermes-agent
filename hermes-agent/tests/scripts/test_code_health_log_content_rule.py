"""HX017: gateway/adapter code must not log user content or identities at INFO+ (#46006)."""

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID, rule_applies
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import _commit, _repo, _verdict


def _flagged(src: str) -> bool:
    return bool(list(CHECKERS["HX017"](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src", [
    "logger.info('Response for %s: %s', chat_id, content[:200])",
    "logger.info('[Email] New message from %s: %s', sender_addr, subject)",
    "logger.warning('Unauthorized user: %s (%s)', source.user_id, source.user_name)",
    "logger.info('msg=%r', (event.text or '')[:80].replace('\\n', ' '))",
    "logger.info('by %s', interaction.user.display_name)",
    "logger.info(f'reply text={text}')",
    "logger.error('body=%s', json.dumps(body, ensure_ascii=False)[:500] if body else None)",
    "self._log.warning('quote=%s', quote_text)",
    "logging.info('from %s', from_number)",
    "logger.info('x %s' % text)",
    "logger.info('Voice input from user %d: %s', user_id, transcript[:100])",
    "logger.info('resolved (user=%s)', user_display)",
    "logger.info('from %s', getattr(message.author, 'display_name', 'unknown'))",
])
def test_flags_user_content_at_info_and_above(src):
    assert _flagged(src), src


@pytest.mark.parametrize("src", [
    "logger.debug('Response text: %s', content)",  # DEBUG is where the raw value belongs
    "logger.info('Response for %s (%d chars)', chat_id, len(content))",
    "logger.info('from %s', redact_phone(from_number))",
    "logger.info('chat %s', hash_for_log(chat_id))",
    "logger.warning('HTTP %s: %s', resp.status, resp.text[:200])",  # the remote API's answer
    "logger.error('stderr=%s err=%s', stderr_text, error_text)",
    "logger.info('user=%s chat=%s', user_id, chat_id)",  # ids are allowed
    "print('text', text)",
    "logger.info('has text: %s', bool(text))",
    "logger.info('from %s', getattr(message.author, 'id', 'unknown'))",
    "logger.warning('HTTP %s', getattr(resp, 'text', ''))",  # transport payload via getattr
])
def test_ignores_metadata_debug_and_transport_payloads(src):
    assert not _flagged(src), src


@pytest.mark.parametrize("path, applies", [
    ("gateway/run_turn.py", True), ("gateway/platforms/webhook.py", True),
    ("plugins/platforms/email/adapter.py", True), ("agent/turn_x.py", False),
    ("tests/gateway/test_x.py", False), ("plugins/platforms/email/tests/test_x.py", False),
])
def test_scope_is_gateway_and_platform_plugins(path, applies):
    assert rule_applies(RULES_BY_ID["HX017"], path) is applies


_LEAK = "import logging\nlogger = logging.getLogger(__name__)\n\n\ndef f(text):\n    logger.info('got %s', text)\n"


@pytest.mark.parametrize("head, code, needle", [
    ({"gateway/x.py": _LEAK}, 1, "hash_for_log"),
    ({"gateway/x.py": _LEAK.replace("text)\n", "text)  # health: allow HX017 -- config string\n")}, 0,
     "config string"),
])
def test_new_hit_blocks_and_waiver_silences(tmp_path, capsys, head, code, needle):
    repo, _ = _repo(tmp_path)
    got, out = _verdict(repo, _commit(repo, {"gateway/y.py": "X = 1\n"}), head, capsys)
    assert got == code and needle in out, out
