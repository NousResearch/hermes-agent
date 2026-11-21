"""A shortened approval preview keeps the commands after a long heredoc (#158342): the body is data, so it is
elided and the step and its target stay in view. The preview still counts as cut (Deny-only unless the whole
command is shown), so nothing here can be approved unread."""

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.base_exec_approval import elide_heredoc_bodies
from gateway.run_turn_runner_approval_text import _cutting_text_prompt

BODY = "\n".join(f"- line {i}: some long issue text that goes on and on" for i in range(200))
STEP = "gh issue create -R owner/repo --title 'Fix it' --body-file /tmp/body.md"
COMMAND = f"cat > /tmp/body.md <<'EOF'\n{BODY}\nEOF\n{STEP}"


class _Chat(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 1000

    def __init__(self, *, send_ok: bool = True) -> None:
        self.sent, self.prompts, self._send_ok = [], [], send_ok

    def _exec_approval_cmd_budget(self, description: str, smart_denied: bool) -> int:
        return 600

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=self._send_ok, message_id="m1")

    async def _send_exec_approval_prompt(self, prompt):
        self.prompts.append(prompt)
        return SendResult(success=True, message_id="card")


class _Cutting(_Chat):
    """A text-only adapter whose send cuts long messages (no split)."""

    splits_long_messages = False
    MAX_MESSAGE_LENGTH = 900

    def max_message_length_for_chat(self, chat_id):
        return self.MAX_MESSAGE_LENGTH

    def message_len_fn_for_chat(self, chat_id):
        return len


def test_elides_only_long_heredoc_bodies_and_keeps_every_command_line():
    shown = elide_heredoc_bodies(COMMAND)
    assert shown.splitlines() == ["cat > /tmp/body.md <<'EOF'", "- line 0: some long issue text that goes on and on",
                                  "- line 1: some long issue text that goes on and on", "…", "EOF", STEP]
    short = "cat > a <<EOF\none\ntwo\nthree\nEOF\nrm a"
    assert elide_heredoc_bodies(short) == short
    assert elide_heredoc_bodies("ls -la && echo '<<not a heredoc'") == "ls -la && echo '<<not a heredoc'"


@pytest.mark.parametrize("command", [
    f"bash <<-EOF\n\t{BODY}\n\tEOF\necho after",  # <<- ends on a tab-indented line
    f"cat <<A <<'B'\n{BODY}\nA\n{BODY}\nB\necho after",  # two heredocs opened on one line, read in order
])
def test_tab_stripped_and_stacked_heredocs_keep_the_following_command(command):
    shown = elide_heredoc_bodies(command)
    assert shown.endswith("echo after") and shown.count("…") == command.count("<<") and len(shown) < 300


def test_a_heredoc_with_no_end_line_is_left_whole():
    unterminated = f"cat > x <<'EOF'\n{BODY}"
    assert elide_heredoc_bodies(unterminated) == unterminated


@pytest.mark.asyncio
async def test_card_preview_shows_the_step_after_a_long_heredoc_and_stays_cut():
    adapter = _Chat()
    await adapter.send_exec_approval("c1", COMMAND, "sk", description="flagged", request_id="r1")
    text = adapter.prompts[-1].text
    assert STEP in text and "line 150" not in text and "Preview shortened" in text
    assert adapter.sent == [f"```\n{COMMAND}\n```"]  # the whole command still goes out before the card


@pytest.mark.asyncio
async def test_card_with_unsendable_full_command_is_still_deny_only():
    adapter = _Chat(send_ok=False)
    await adapter.send_exec_approval("c1", COMMAND, "sk", description="flagged", request_id="r1")
    assert STEP in adapter.prompts[-1].text
    assert [choice for _, choice, _ in adapter.prompts[-1].actions] == ["deny"]


def test_cutting_text_prompt_shows_the_step_and_offers_deny_only():
    text, deny_only = _cutting_text_prompt(_Cutting(), "c1", COMMAND, "flagged", "/")
    assert deny_only and STEP in text and "line 150" not in text and len(text) <= _Cutting.MAX_MESSAGE_LENGTH
