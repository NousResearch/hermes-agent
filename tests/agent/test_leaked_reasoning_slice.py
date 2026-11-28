"""Regression tests for the leaked-reasoning-slice suppression.

Every (content, reasoning) pair below reproduces a shape observed when an
OpenAI-compatible relay flushes part of the thinking buffer into the content
channel on tool-call turns: ``content`` is a contiguous mid-word cut of the
reasoning. The negatives reproduce real preambles. The fixture texts are
sanitized structural equivalents of the observed instances — same lengths,
punctuation, and substring relationships — not verbatim transcripts.
"""
from types import SimpleNamespace

from agent.chat_completion_helpers import strip_leaked_reasoning_slice, build_assistant_message


class _MockAgent:
    """Just enough of AIAgent for build_assistant_message."""
    verbose_logging = False
    reasoning_callback = None
    stream_delta_callback = None
    _stream_callback = None

    def _extract_reasoning(self, m):
        return getattr(m, "reasoning_content", None)

    def _strip_think_blocks(self, t):
        return t

    def _needs_thinking_reasoning_pad(self):
        return False

    def _split_responses_tool_id(self, raw):
        return raw, None

    def _deterministic_call_id(self, name, args, i):
        return f"call_{i}"

    def _derive_responses_function_call_id(self, call_id, rid):
        return rid or call_id


def _msg(content, reasoning, tool_calls=None):
    return SimpleNamespace(content=content, reasoning_content=reasoning,
                           reasoning=None, tool_calls=tool_calls)


TOOL = [SimpleNamespace(id="1", call_id="1", type="function",
                        function=SimpleNamespace(name="execute_code", arguments="{}"))]

# Leaked-fragment shapes (tool-call turns, mid-word cuts of the reasoning).
FRAGMENTS = [
    ('读正文',
     '用户想读正文，不想只看摘要。我需要先确认文章的完整链接，再调用抓取工具获取页面内容。抓取之前先检查是否需要登录：如果页面有付费墙，直接抓取只会拿到标题和引导语，那样读正文就失败了。拿到正文后，先看总长度：如果超过单次处理上限，就分段读取再拼接，避免中间截断丢失段落。还要注意编码问题：有些站点返回的内容里混着转义字符，需要先清洗再统计字数。最后把要点整理成中文回复，引用原文时保留原始措辞，不添加自己的转述。整理完毕后再输出最终结论，确保没有遗漏关键信息点。'),
    ('二 (初稿)\n   -',
     '整理一下手头的材料。计划分两步走：第一步核对原文，第二步审阅材料二 (初稿)\n   - 先列出所有引用了原文数字的段落\n   - 再逐条比对数字是否一致\n   - 有出入的段落标记出来，等会儿集中处理\n第二步完成后再决定是否需要重写结论段。审阅时注意区分两类问题：一类是事实错误，必须改；另一类只是表述不够清楚，可以保留原意只调整说法。如果初稿的整体结构没问题，就只做局部修订，不推倒重来。修订完成后把改动记录单独列一页，方便前后对比，也让读者一眼看出哪些地方动过、哪些地方保持原样。'),
    ('slot gpt-5-min',
     'The routing table needs a second look before I change anything. The config maps the reviewer slot gpt-5-mini to the small pool and the writer slot to the large one, which is the opposite of what the docs describe. If the logs show review passes landing on the small pool, that is a real routing bug; if they show the writer there, the docs are simply stale. I should also check the fallback chain: a typo in a fallback name would silently drop the backup and only surface during an outage, which is the worst time to find out. Plan: read the config, compare the declared slots against the last hour of logs, then propose the minimal edit. Let me start with the config file and keep the log query ready.'),
    ('m53_',
     'Before writing the summary I need the checkpoint layout. The runner writes each pass to a numbered folder: m52_ holds the baseline, m53_ is the current pass, and m54_ will appear once this one is accepted. The summary should therefore quote m53_ numbers, not the baseline, and mention the delta against m52_. I also need the timestamps: if m53_ was written before the last config change, the numbers are stale and the pass must be rerun before anyone cites them. Let me list the directory, check the newest folder first, and only then draft the summary paragraph with the comparison table.'),
    ('向公益基金',
     '这一段讲的是资金流向，需要仔细核对。原文说该笔款项向公益基金划转了三次，每次金额递减，但表格里只列了两行。先查附录：如果附录补了第三行，正文就是漏印；如果附录也只有两行，那正文的说法就站不住。还要注意口径：向公益基金划转和通过公益基金再分配是两回事，前者是直接转账，后者多一层中转，不能混在一个总数里。核对时把每个数字的页码记下来，回复里引用原文时标注页码，方便读者回查。如果两处数字对不上，以附录为准，并在回复里说明差异。最后再检查一遍单位：原文用的是万元，别在换算时弄错。'),
    ('加总: 2,656',
     '把三个分项的数字重新验算一遍，防止抄错。先看前两项：加总: 2,656 = 1,204 + 1,452，两个分项之和与正文合计一致，说明这两项没问题。第三个分项 986 还没核：如果它也该计入合计，那合计应该是 3,642，和正文的 2,656 对不上，需要回原文确认第三个分项是不是独立列示。另外注意单位是否一致：两个分项用千，一个用万，直接相加就会错一个数量级。把每一步的算式都写在笔记里，回复时附上验算过程，让读者可以复算。如果最终发现是原文印错，就在回复里标注勘误建议，而不是直接改数字。'),
    ('3274, mtime 02:01:',
     'Confirm the reload actually picked up the new file. The log says the config was rewritten at line 3274, mtime 02:01:16, while the service reports a start time of 02:01:51 — thirty-five seconds later, so the process must have loaded the new bytes. That is the passive check. The active check is behavioral: send one request that exercises the changed branch and watch the response. If the branch answers with the new field, the reload is proven; if it answers with the old shape, the start time was misleading and the service needs a hard restart. Run the probe now and record both timestamps in the report so the next person can see the evidence, not just the conclusion.'),
    ('started after mtime, it must',
     'The user wants proof the fix is live after the restart, not just a log line. Verification approach: first confirm the service restarted and loaded the patched file — new PID, start time later than the file mtime. Since the module is imported once at startup and the process started after mtime, it must hold the new code; that is sound but indirect. The direct proof is behavioral: exercise the changed path and observe the new behavior. I cannot force the upstream condition on demand, so the best available evidence is a focused simulation that drives the real code path with a fake stream: emit the malformed chunks the relay produced and assert the output stays clean. That is an integration test of the patched module, not just the helper function. There may already be a harness for stream handling in the test suite — check how the existing retry tests construct the agent and reuse their fakes instead of building a new one from scratch. If the simulation passes and the timestamps line up, report both; if either check fails, escalate to a hard restart before drawing any conclusions about the fix itself.'),
]

# Real-preamble shapes that must survive the raw filter. ('无新文' — a stop-turn reply
# that IS a substring of its own reasoning — is covered by the gate test below:
# the tool_calls gate, not the raw filter, is what protects stop-turn replies.)
KEPT = [
    ('Last run failed with 1 failed test. Let me find which test failed and its error context.',
     'The last run failed with 1 failed test. Let me find which test failed. The results directory has the error context; there are also modified files in the working tree from the earlier fix attempt. Let me check the failed test first and read its stack trace, then decide whether the failure is related to those edits or independent. If it is the same assertion as before, the earlier fix did not land and needs a second look.'),
    ("I'll read the SKILL.md attachment and fetch the article",
     "The user wants me to research an article URL with a SKILL.md as the guide. Let me read both first. The skill file sits outside the workspace but reading it should still work — the warning only notes the boundary. Then fetch the URL: no direct web fetch tool is listed, but the browser tool can navigate and extract. Batch the two reads, then summarize the article against the skill's checklist."),
    ('直接编辑示例应用的 package.json 添加依赖（recharts 3.9.2、@heroicons/react 2.2.0，已在本地缓存）',
     'recharts 3.9.2 和 @heroicons/react 2.2.0 已经在根 node_modules 里（可能是某个依赖提升上来的），但示例应用自己的 package.json 没有声明。先确认版本号没记错，再直接在示例应用的 package.json 里补上这两个依赖，然后跑一次安装——因为包已经在本地缓存，应该很快。注意 workspace 的严格模式：每个应用必须自己声明依赖，靠提升不可靠。改完先读一遍文件确认 JSON 没写坏。'),
]

STOP_TURN_SUBSTRING_REPLY = ('无新文', '两路采集都返回空：第一路 0 条新内容，第二路也是 0 条，最新一条还是昨天报过的那条，时间戳在上一轮运行之前。连续性记录被截断过，但按时间推算，上一轮已经覆盖了它；另一路最新的一条更是三天前的旧闻。把两路的检查时间都记进日志，方便下一轮对比。没有需要推送的内容，按约定回复无新文三个字即可，不要展开成段落，也不要重复旧条目。')


def test_real_fragments_are_suppressed():
    for content, reasoning in FRAGMENTS:
        out = strip_leaked_reasoning_slice(content, reasoning)
        assert out == "", f"fragment {content!r} was not suppressed"


def test_real_preambles_are_kept():
    for content, reasoning in KEPT:
        out = strip_leaked_reasoning_slice(content, reasoning)
        assert out == content, f"real content {content!r} was wrongly suppressed"


def test_sentence_aligned_substring_preamble_is_kept():
    # A real preamble that is a contiguous substring of its reasoning, but a
    # complete sentence — the sentence-end guard must spare it.
    content = "Let me check where playwright got installed."
    reasoning = (
        "The test imports playwright from the node_modules directory but the CLI is not on "
        "PATH. Let me check where playwright got installed. The install log said it went to "
        "the user-level bin, so I need the full path to invoke it from the test script."
    )
    assert content in reasoning
    assert strip_leaked_reasoning_slice(content, reasoning) == content


def test_quote_ending_fragment_is_suppressed():
    # A fragment shape that ends with a quote, not a sentence end.
    reasoning = (
        "The user reports the assistant messages contain random word fragments. The symptom "
        "is 随机切分词\" appearing as standalone replies. I need to check whether the relay "
        "mirrors the reasoning channel into content with different chunk boundaries."
    )
    assert strip_leaked_reasoning_slice("切分词\"", reasoning) == ""


def test_guards():
    assert strip_leaked_reasoning_slice("ok", "x" * 500) == "ok"          # too short
    assert strip_leaked_reasoning_slice("读正文", "x" * 100) == "读正文"    # reasoning < 200
    assert strip_leaked_reasoning_slice("a good preamble", "short") == "a good preamble"  # ratio
    assert strip_leaked_reasoning_slice("完全无关的内容", "x" * 500) == "完全无关的内容"   # not substring
    assert strip_leaked_reasoning_slice(None, "x" * 500) is None           # non-string safe


def test_build_assistant_message_gate():
    """Suppression fires only on tool-call turns — a stop turn keeps its reply."""
    agent = _MockAgent()
    content, reasoning = FRAGMENTS[0]

    # tool-call turn: the leaked slice is dropped from the persisted row
    msg = build_assistant_message(agent, _msg(content, reasoning, tool_calls=TOOL), "tool_calls")
    assert msg["content"] == ""
    assert msg["reasoning"] == reasoning

    # stop turn: identical text would be a real (short) reply and must survive
    msg = build_assistant_message(agent, _msg(content, reasoning, tool_calls=None), "stop")
    assert msg["content"] == content

    # '无新文': real cron reply, a substring of its own reasoning — the gate keeps it
    msg = build_assistant_message(agent, _msg(*STOP_TURN_SUBSTRING_REPLY, tool_calls=None), "stop")
    assert msg["content"] == STOP_TURN_SUBSTRING_REPLY[0]


# A spliced flush (words dropped between thinking fragments) that no
# contiguous-substring rule can match.
SPLICED_LEAK = (
    'the previous turn, or so I believe. Hmm — I don\'t see it in my earlier notes; '
    'let me re-check the work log. The log covers the last thirty'
)
SPLICED_LEAK_REASONING = (
    "The user asked for a progress summary. The newest message requests the current task "
    "list. The permission step was already finished in the previous turn, or so I believe. "
    "Hmm, wait — I don't see it in my earlier notes; let me re-check the work log. The log "
    "covers the last thirty turns or so, including the setup reads. It mentions the "
    "permission grant near the end. Let me query the task table before answering, then "
    "summarize what is still open."
)


def test_spliced_leak_without_relay_flag_is_untouched_by_14():
    """Documents the contiguous-slice boundary: a splice is not a contiguous substring,
    so the text rule alone cannot catch it — the structural flag is what suppresses it."""
    agent = _MockAgent()
    out = build_assistant_message(
        agent, _msg(SPLICED_LEAK[0], SPLICED_LEAK_REASONING, tool_calls=TOOL), "tool_calls")
    assert out["content"] == SPLICED_LEAK[0]


def test_mirror_relay_flag_strips_spliced_tool_turn_content():
    """Once streaming has flagged the relay (same-chunk mirroring or content interleaved
    with reasoning), tool-turn content is thinking-buffer material and is stripped from
    storage — splice or not."""
    agent = _MockAgent()
    agent._reasoning_mirror_relay = True
    out = build_assistant_message(
        agent, _msg(SPLICED_LEAK[0], SPLICED_LEAK_REASONING, tool_calls=TOOL), "tool_calls")
    assert out["content"] == ""

    # stop turns are untouched by the flag: the reply may be short but it is real
    out = build_assistant_message(
        agent, _msg(SPLICED_LEAK[0], SPLICED_LEAK_REASONING, tool_calls=None), "stop")
    assert out["content"] == SPLICED_LEAK[0]
