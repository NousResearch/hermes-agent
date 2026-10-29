"""Localise the lifecycle/warning lines emitted through ``_emit_status`` / ``_emit_warning``.

Every one of those lines passes a single funnel in ``agent.status_output``:
``_emit_status_kind`` (CLI print + ``status_callback``), and the translation happens at the
funnel's exit. The internal English constants are left untouched, so nothing downstream that
matches on English has to change — the gateway's noise filters and
``is_compaction_progress_status`` keep working verbatim, and an English user
(``HERMES_LANGUAGE=en``) sees byte-identical output.

The catalog uses the English source line as its single source of truth: ``{name}`` compiles to a
named group for a whole-string match, and the captured values are filled back into the
translation. Adding or changing a line therefore touches exactly one place, and the two sides
can never drift apart. Translations live in this table (one column per language, see
``_LANG_COLUMN``) rather than in ``locales/<lang>.yaml``, because a new key there would have to be
translated into all sixteen shipped languages at once.

``normalize_status`` goes the other way (localised -> English) for the downstream checks that
still match on English.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# key -> (English source line, Chinese translation)
STATUS_TEXTS: dict[str, tuple[str, str]] = {'status.compaction.running': ['🗜️ Compacting context — summarizing earlier conversation so I can continue...',
                               '🗜️ 正在压缩上下文 — 正在摘要之前的对话，以便继续…'],
 'status.compaction.heartbeat': ['🗜️ Compacting context — still summarizing earlier conversation so I can '
                                 'continue...',
                                 '🗜️ 正在压缩上下文 — 仍在摘要之前的对话，以便继续…'],
 'status.compaction.done': ['✓ Context compaction complete — continuing turn...', '✓ 上下文压缩完成 — 继续本轮对话…'],
 'status.compaction.pre_api': ['📦 Pre-API compression: ~{tokens} tokens near the context/output limit. Compacting '
                               'before the next model call.',
                               '📦 请求前压缩：约 {tokens} tokens 接近上下文/输出上限。在下次调用模型前先压缩。'],
 'status.compaction.preflight': ['📦 Preflight compression: ~{tokens} tokens >= {threshold} threshold. This may take '
                                 'a moment.',
                                 '📦 预压缩：约 {tokens} tokens ≥ 阈值 {threshold}。可能需要稍等片刻。'],
 'status.compaction.idle': ['💤 Resumed after {idle_seconds}s idle — compacting ~{tokens} tokens before continuing.',
                            '💤 空闲 {idle_seconds} 秒后恢复 — 先压缩约 {tokens} tokens 再继续。'],
 'status.compaction.too_large': ['🗜️ Context too large (~{tokens} tokens) — compressing ({attempt}/{cap})...',
                                 '🗜️ 上下文过大（约 {tokens} tokens）— 正在压缩（{attempt}/{cap}）…'],
 'status.compaction.messages': ['🗜️ Compressed {before} → {after} messages, retrying...',
                                '🗜️ 已压缩 {before} → {after} 条消息，正在重试…'],
 'status.compaction.tokens': ['🗜️ Compressed ~{before} → ~{after} tokens, retrying...',
                              '🗜️ 已压缩 ~{before} → ~{after} tokens，正在重试…'],
 'status.compaction.reduced': ['🗜️ Context reduced to {new_ctx} tokens (was {old_ctx}), retrying...',
                               '🗜️ 上下文已缩减至 {new_ctx} tokens（原 {old_ctx}），正在重试…'],
 'status.compression.count_warning': ['⚠️  Session compressed {count} times — accuracy may degrade. Consider /new to '
                                      'start fresh.',
                                      '⚠️  本会话已压缩 {count} 次 — 回答准确度可能下降。建议用 /new 开启新会话。'],
 'status.overflow.blocked': ['⚠ Context is over the compression threshold (~{tokens} tokens >= {threshold}) but '
                             'compression is currently blocked ({reason}). The model may stop responding. Run /new '
                             'to start a fresh session or /compress to retry immediately.',
                             '⚠ 上下文已超过压缩阈值（约 {tokens} tokens ≥ {threshold}），但压缩当前被阻塞（{reason}）。模型可能停止响应。请用 /new '
                             '开启新会话，或用 /compress 立即重试。'],
 'status.overflow.compression_disabled': ['⚠️ Session context (~{preflight_tokens} tokens) exceeds the model context '
                                          'window (~{context_length} tokens) with compression disabled '
                                          '(compression.enabled: false). Use /compact to compress history or enable '
                                          'compression in config.yaml.',
                                          '⚠️ 本会话上下文（约 {preflight_tokens} tokens）已超出模型上下文窗口（约 {context_length} '
                                          'tokens），而压缩已禁用（compression.enabled: false）。请用 /compact 压缩历史，或在 '
                                          'config.yaml 中启用压缩。'],
 'status.compression.no_aux_provider': ['⚠ No auxiliary LLM provider configured: Hermes has no helper model for '
                                        'summarising long chats, so older messages will be cut without a summary. '
                                        'Run `hermes setup` to add one.',
                                        '⚠ 未配置辅助 LLM 提供方：Hermes 没有用于摘要长对话的辅助模型，较早的消息将被直接截断而无摘要。请运行 `hermes setup` '
                                        '添加。'],
 'status.compression.aux_provider_unavailable': ["⚠ Configured auxiliary compression provider '{provider}' is "
                                                 'unavailable, so older messages in long chats will be cut without a '
                                                 'summary. Sign in to that provider again, or change '
                                                 'auxiliary.compression in your config.',
                                                 '⚠ 已配置的压缩提供方「{provider}」当前不可用，长对话中较早的消息将被直接截断而无摘要。请重新登录该提供方，或修改配置中的 '
                                                 'auxiliary.compression。'],
 'status.compression.aux_model_fellback': ["ℹ Configured compression model '{model}' failed, so Hermes summarised "
                                           'with your main model instead. Check auxiliary.compression.model in your '
                                           'config.',
                                           'ℹ 已配置的压缩模型「{model}」调用失败，已改用主模型完成摘要。请检查配置中的 auxiliary.compression.model。'],
 'status.compression.summary_failed': ['⚠ Compression summary failed: {error}. Inserted a fallback context marker.',
                                       '⚠ 压缩摘要失败：{error}。已插入兜底上下文标记。'],
 'status.compression.aborted': ['⚠ Compression aborted: {error}. No messages were dropped — conversation continues '
                                'unchanged. Run /compress to retry, or /new to start a fresh session.',
                                '⚠ 压缩已中止：{error}。没有丢弃任何消息 — 对话保持不变。可运行 /compress 重试，或用 /new 开启新会话。'],
 'status.compression.empty_transcript': ['⚠ Compression returned an empty transcript. No session split was '
                                         'performed; conversation continues unchanged.',
                                         '⚠ 压缩返回了空对话记录。未执行会话切分；对话保持不变。'],
 'status.compression.concurrent_skip': ['⚠ Skipping concurrent compression — another path is already compressing '
                                        'this session. Will retry after it finishes.',
                                        '⚠ 跳过并发压缩 — 已有另一个流程正在压缩本会话。待其完成后重试。'],
 'status.compression.refused_would_grow': ['⚠️ Compression refused: the generated summary would have GROWN the '
                                           'conversation instead of shrinking it. No messages were dropped — '
                                           'conversation continues unchanged.',
                                           '⚠️ 压缩被拒绝：生成的摘要会让对话变得更长而非更短。没有丢弃任何消息 — 对话保持不变。'],
 'status.compression.codex_failed': ['⚠ Codex app-server compaction failed: {error}',
                                     '⚠ Codex app-server 压缩失败：{error}'],
 'status.model.primary_restored': ['✅ Primary model restored: {model} via {provider}; fallback {previous_model} via '
                                   '{previous_provider} is no longer active.',
                                   '✅ 已恢复主模型：{model}（经 {provider}）；兜底模型 {previous_model}（经 '
                                   '{previous_provider}）已不再生效。'],
 'status.model.not_entitled': ['🚫 This account is not entitled to {model} via {provider}; it will be skipped until '
                               'restart. Switch to an entitled model via /model or `hermes model`.',
                               '🚫 当前账号无权使用 {model}（经 {provider}）；重启前将跳过该模型。请用 /model 或 `hermes model` 切换到有权限的模型。'],
 'status.empty.no_content_fallback': ['❌ Model returned no content after all retries and fallback attempts.',
                                      '❌ 重试并尝试兜底后，模型仍未返回任何内容。'],
 'status.empty.no_content': ['❌ Model returned no content after all retries. No fallback providers configured.',
                             '❌ 重试后模型仍未返回任何内容。未配置兜底提供方。'],
 'status.empty.reasoning_only': ['⚠️ Model produced reasoning but no visible response after all retries. Returning '
                                 'empty.',
                                 '⚠️ 模型只产出了推理过程，重试后仍无可见回答。返回空内容。'],
 'status.empty.stream_interrupted': ['↻ Stream interrupted — using delivered content as final response',
                                     '↻ 流式中断 — 已把送出的内容作为最终回答'],
 'status.empty.after_tools': ['↻ Empty response after tool calls — using earlier content as final answer',
                              '↻ 工具调用后返回空 — 已把先前内容作为最终回答'],
 'status.empty.after_tools_nudge': ['⚠️ Model returned empty after tool calls — nudging to continue',
                                    '⚠️ 模型在工具调用后返回空 — 已提示其继续'],
 'status.empty.switching_fallback': ['⚠️ Model returning empty responses — switching to fallback provider...',
                                     '⚠️ 模型持续返回空 — 正在切换到兜底提供方…'],
 'status.empty.switched_fallback': ['↻ Switched to fallback: {model} ({provider})', '↻ 已切换到兜底：{model}（{provider}）'],
 'status.empty.retrying': ['⚠️ Empty response from model — retrying ({n}/{budget}) in {wait}s{note}',
                           '⚠️ 模型返回空 — {wait} 秒后重试（{n}/{budget}）{note}'],
 'status.empty.cost_estimate': ['ℹ️ Estimated cost of these empty attempts: ~${cost} (input tokens are billed per '
                                'attempt even when no answer is produced)',
                                'ℹ️ 这些空回答预计花费：约 ${cost}（即使没有产出回答，每次尝试的输入 token 仍会计费）'],
 'status.empty.thinking_prefill': ['↻ Thinking-only response — prefilling to continue ({n}/{budget})',
                                   '↻ 只返回了思考内容 — 预填充以继续（{n}/{budget}）'],
 'status.empty.skip_retries': ['⚠️ Model is repeatedly returning empty content — skipping further retries to avoid '
                               'repeat charges',
                               '⚠️ 模型反复返回空内容 — 为避免重复计费，跳过后续重试'],
 'status.error.max_retries': ['❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.',
                              '❌ 无效响应重试次数已达上限（{max_retries}）。放弃。'],
 'status.error.no_answer': ["❌ The model provider didn't answer after all retries. Send /retry, or switch models "
                            'with /model.',
                            '❌ 重试完毕后模型提供方仍无响应。请发送 /retry，或用 /model 切换模型。'],
 'status.error.ollama_context': ['❌ Ollama runtime context is too small for Hermes tool use',
                                 '❌ Ollama 运行时上下文过小，无法支撑 Hermes 的工具调用'],
 'status.error.billing': ['❌ Billing or credits exhausted — {summary}', '❌ 余额或额度已耗尽 — {summary}'],
 'status.error.rate_limited': ['❌ Rate limited after {n} retries — {summary}', '❌ 重试 {n} 次后仍被限流 — {summary}'],
 'status.error.api_failed': ['❌ API failed after {n} retries — {summary}', '❌ 重试 {n} 次后 API 仍失败 — {summary}'],
 'status.error.billing_unverified': ['❌ Provider reported usage/credit exhaustion (unverified — may be a '
                                     'content-filter rejection) — {summary}',
                                     '❌ 提供方报告用量/额度耗尽（未验证 — 也可能是内容过滤拒绝）— {summary}'],
 'status.tool.guardrail': ['⚠️ Tool guardrail halted {tool}: {code}', '⚠️ 工具护栏已中止 {tool}：{code}'],
 'status.tool.no_call': ['↻ Model signaled a tool call but sent none — re-prompting ({n}/3)',
                         '↻ 模型表示要调用工具却没给出调用 — 正在重新提示（{n}/3）'],
 'status.filter.terminated': ['Content filter terminated stream; switching to fallback...', '内容过滤器终止了流式输出；正在切换到兜底…'],
 'status.filter.refusal': ['⚠️ The model declined to respond to this request (safety refusal).',
                           '⚠️ 模型拒绝回答该请求（安全拒答）。'],
 'status.turn.budget_exhausted': ['⚠️ Iteration budget exhausted ({used}/{max}) — asking model to summarise',
                                  '⚠️ 迭代次数已用尽（{used}/{max}）— 正在让模型做总结'],
 'status.turn.stale_connections': ['🔌 Detected stale connections from a previous provider issue — cleaned up '
                                   'automatically. Proceeding with fresh connection.',
                                   '🔌 检测到上次提供方故障遗留的失效连接 — 已自动清理。正在使用新连接继续。'],
 'status.turn.stalled': ['⚠️ This turn stopped making progress ({idle}s without activity); attempting recovery so '
                         'the session can continue.',
                         '⚠️ 本轮已停止推进（{idle} 秒无活动）；正在尝试恢复，以便会话继续。'],
 'status.turn.aborted_watchdog': ['⚠️ Turn aborted by the liveness watchdog ({idle}s without activity); lease '
                                  'renewal stopped so the session can be reclaimed. You can retry your message.',
                                  '⚠️ 本轮已被存活看门狗中止（{idle} 秒无活动）；已停止续租，会话可被回收。你可以重新发送消息。'],
 'status.stream.empty_keepalive': ['⚠️ Provider stream returned an empty keepalive frame — retrying this turn '
                                   'without streaming (streaming stays off for this session).',
                                   '⚠️ 提供方流式返回了空的保活帧 — 本轮改用非流式重试（本会话将保持关闭流式）。'],
 'status.session.waiting': ['⏳ Another Hermes process is using this session; waiting for it to finish before '
                            'starting your turn...',
                            '⏳ 另一个 Hermes 进程正在使用本会话；等待其完成后再开始你的这一轮…'],
 'status.session.still_waiting': ['⏳ Still waiting for the other Hermes process on this session ({elapsed}s)...',
                                  '⏳ 仍在等待占用本会话的另一个 Hermes 进程（{elapsed} 秒）…'],
 'status.session.free': ['Session is free; loading the latest transcript...', '会话已空闲；正在加载最新对话记录…'],
 'status.kanban.nudge': ['⚠️ Kanban worker tried to exit without kanban_complete/kanban_block — nudging to finish',
                         '⚠️ 看板工作进程试图在未调用 kanban_complete/kanban_block 的情况下退出 — 已提示其完成任务'],
 'status.memory.recalled_one': ['{glyph} {provider} — recalled 1 memory', '{glyph} {provider} — 召回 1 条记忆'],
 'status.memory.recalled_many': ['{glyph} {provider} — recalled {count} memories',
                                 '{glyph} {provider} — 召回 {count} 条记忆'],
 'status.memory.recalled_vague': ['{glyph} {provider} — recalled relevant memory', '{glyph} {provider} — 召回了相关记忆'],
 'status.provider.no_response_stream': ['⚠️ No response from provider for {elapsed}s (model: {model}, context: '
                                        '~{tokens} tokens). Reconnecting...',
                                        '⚠️ 提供方已 {elapsed} 秒无响应（模型：{model}，上下文：约 {tokens} tokens）。正在重连…'],
 'status.provider.no_response_nonstream': ['⚠️ No response from provider for {elapsed}s (non-streaming, model: '
                                           '{model}). {hint}',
                                           '⚠️ 提供方已 {elapsed} 秒无响应（非流式，模型：{model}）。{hint}'],
 'status.provider.bedrock_no_events': ['⚠️ No events from Bedrock for {stale}s (model: {model}). Aborting...',
                                       '⚠️ Bedrock 已 {stale} 秒没有任何事件（模型：{model}）。正在中止…'],
 'status.provider.codex_no_first_event': ['⚠️ No first stream event from provider in {elapsed}s (codex stream, '
                                          'model: {model}). Reconnecting.{silent}',
                                          '⚠️ 提供方在 {elapsed} 秒内没有发出首个流式事件（codex 流，模型：{model}）。正在重连。{silent}'],
 'status.provider.codex_stream_idle': ['⚠️ Codex stream sent no events for {stale}s after {arm_point} (model: '
                                       '{model}). Reconnecting.',
                                       '⚠️ Codex 流在 {arm_point} 之后 {stale} 秒没有事件（模型：{model}）。正在重连。'],
 'status.provider.wait_notice': ['⚠ no response from provider in {elapsed}s — reconnecting...',
                                 '⚠ 提供方已 {elapsed} 秒无响应 — 正在重连…'],
 'status.overflow.payload_413': ['⚠️  Request payload too large (413) — compression attempt {attempt}/{cap}...',
                                 '⚠️  请求体过大（413）— 正在压缩（第 {attempt}/{cap} 次）…'],
 'status.overflow.compressed_payload': ['🗜️ Compressed {before} → {after} payload bytes, retrying...',
                                        '🗜️ 已把请求体压缩为 {before} → {after} 字节，正在重试…'],
 'status.overflow.cannot_shrink': ['📐 Compression could not reduce the request further — removed retained vision '
                                   'payloads and retrying...',
                                   '📐 压缩已无法进一步缩小请求 — 已移除保留的视觉载荷并重试…'],
 'status.overflow.output_cap_too_large': ['⚠️  Output cap too large for current prompt — retrying with '
                                          'max_tokens={safe_out} (provider_available={available_out}, '
                                          'estimated_request_tokens={request_tokens}; context_length unchanged at '
                                          '{ctx})',
                                          '⚠️  输出上限相对当前提示词过大 — 改用 max_tokens={safe_out} '
                                          '重试（提供方上限={available_out}，预估请求 tokens={request_tokens}；context_length 保持 '
                                          '{ctx} 不变）'],
 'status.overflow.context_limit_from_api': ['Context limit detected from API: {new_ctx} tokens (was {old_ctx})',
                                            '已从 API 探测到上下文上限：{new_ctx} tokens（原为 {old_ctx}）'],
 'status.overflow.using_provider_limit': ['⚠️  Context length exceeded — using provider limit: {old_ctx} → {new_ctx} '
                                          'tokens',
                                          '⚠️  已超出上下文长度 — 改用提供方上限：{old_ctx} → {new_ctx} tokens'],
 'status.overflow.no_max_reported': ['⚠️  Context length exceeded, but provider did not report a max context length; '
                                     'keeping context_length at {old_ctx} tokens and compressing.',
                                     '⚠️  已超出上下文长度，但提供方没有报告最大上下文长度；保持 context_length 为 {old_ctx} tokens 并继续压缩。'],
 'status.overflow.overflow_amount_only': ['Provider reported overflow amount only; keeping context_length at '
                                          '{old_ctx} tokens and compressing.',
                                          '提供方只报告了溢出量；保持 context_length 为 {old_ctx} tokens 并继续压缩。'],
 'status.response.empty_malformed': ['⚠️ Empty/malformed response — switching to fallback...',
                                     '⚠️ 响应为空或格式错误 — 正在切换到兜底…'],
 'status.response.invalid_api': ['⚠️  Invalid API response (attempt {n}/{m}): {details}',
                                 '⚠️  无效的 API 响应（第 {n}/{m} 次）：{details}'],
 'status.response.provider_line': ['   🏢 Provider: {provider}', '   🏢 提供方：{provider}'],
 'status.response.provider_message': ['   📝 Provider message: {message}', '   📝 提供方消息：{message}'],
 'status.response.retrying_in': ['⏳ Retrying in {wait}s ({hint})...', '⏳ {wait} 秒后重试（{hint}）…'],
 'status.response.max_retries_fallback': ['⚠️ Max retries ({n}) for invalid responses — trying fallback...',
                                          '⚠️ 无效响应重试已达上限（{n}）— 正在尝试兜底…'],
 'status.fallback.trying': ['⚠️ {label} — trying fallback...', '⚠️ {label} — 正在尝试兜底…'],
 'status.fallback.max_retries_exhausted': ['⚠️ Max retries ({n}) exhausted — trying fallback...',
                                           '⚠️ 重试次数已达上限（{n}）— 正在尝试兜底…'],
 'status.auth.copilot_reexchange': ['🔐 Copilot credential re-exchanged after model_not_available 400. Retrying '
                                    'request...',
                                    '🔐 Copilot 凭据已重新交换（model_not_available 400）。正在重试请求…'],
 'status.auth.copilot_refreshed_401': ['🔐 Copilot credentials refreshed after 401. Retrying request...',
                                       '🔐 Copilot 凭据已在 401 后刷新。正在重试请求…'],
 'status.auth.refreshed_401': ['🔐 {label} auth refreshed after 401. Retrying request...',
                               '🔐 {label} 鉴权已在 401 后刷新。正在重试请求…'],
 'status.auth.authentication_failed': ['🔐 Authentication failed and could not be refreshed — switching to fallback '
                                       'provider...',
                                       '🔐 鉴权失败且无法刷新 — 正在切换到兜底提供方…'],
 'status.overflow.anthropic_long_context_tier': ['⚠️  Anthropic long-context tier requires extra usage — reducing '
                                                 'context: {old_ctx} → {cap} tokens',
                                                 '⚠️  Anthropic 长上下文档位需要额外额度 — 正在缩减上下文：{old_ctx} → {cap} tokens'],
 'status.recovery.surrogate_stripped': ['⚠️  Stripped invalid surrogate characters from messages. Retrying...',
                                        '⚠️  已清除消息中的无效代理字符。正在重试…'],
 'status.recovery.surrogate_encoding_error': ['⚠️  Surrogate encoding error — retrying after full-payload '
                                              'sanitization...',
                                              '⚠️  代理编码错误 — 已在整包净化后重试…'],
 'status.recovery.stream_interrupted_toolcall_retry': ['⚠️  Stream interrupted mid tool-call — retrying ({n}/4)...',
                                                       '⚠️  工具调用中途流式中断 — 正在重试（{n}/4）…'],
 'status.recovery.truncated_toolcall': ['⚠️  Truncated tool call detected — retrying API call ({n}/4)...',
                                        '⚠️  检测到工具调用被截断 — 正在重试 API 调用（{n}/4）…'],
 'status.recovery.refusal_fallback': ['⚠️ Model declined to respond (safety refusal) — trying fallback...',
                                      '⚠️ 模型拒绝回答（安全拒答）— 正在尝试兜底…'],
 'status.recovery.scratchpad_incomplete': ['⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)',
                                           '⚠️  检测到未闭合的 <REASONING_SCRATCHPAD>（开启后从未闭合）'],
 'status.recovery.retrying_api_call_2': ['🔄 Retrying API call ({n}/2)...', '🔄 正在重试 API 调用（{n}/2）…'],
 'status.recovery.retrying_api_call_3': ['🔄 Retrying API call ({n}/3)...', '🔄 正在重试 API 调用（{n}/3）…'],
 'status.tools.unknown_tool_batch': ["⚠️  Unknown tool '{tool}' in batch — erroring that call, executing {n} valid "
                                     'call(s)',
                                     '⚠️  批次中出现未知工具「{tool}」— 该调用返回错误，其余 {n} 个有效调用照常执行'],
 'status.tools.unknown_tool': ["⚠️  Unknown tool '{tool}' — sending error to model for agent-correction ({n}/3)",
                               '⚠️  未知工具「{tool}」— 已把错误发回模型让它自己纠正（{n}/3）'],
 'status.tools.invalid_json_args': ["⚠️  Invalid JSON in tool call arguments for '{tool}': {error}",
                                    '⚠️  工具调用参数不是合法 JSON（「{tool}」）：{error}'],
 'status.tools.injecting_recovery': ['⚠️  Injecting recovery tool results for invalid JSON...',
                                     '⚠️  正在为无效 JSON 注入恢复用的工具结果…'],
 'status.tools.truncated_refuse': ['⚠️  Truncated tool call response detected again — refusing to execute incomplete '
                                   'tool arguments.',
                                   '⚠️  再次检测到工具调用响应被截断 — 拒绝执行不完整的工具参数。'],
 'status.error.scratchpad_partial': ['❌ Max retries (2) for incomplete scratchpad. Saving as partial.',
                                     '❌ 未闭合 scratchpad 重试已达上限（2 次）。按部分内容保存。'],
 'status.session.lease_timeout': ['⏳ Another Hermes process kept this session busy too long. Your message was not '
                                  'processed - wait for the other process to finish, then send it again.',
                                  '⏳ 另一个 Hermes 进程占用本会话太久，你的消息没有被处理 — 等那个进程结束后再发一次。'],
 'status.local.context_window_grown': ['📈 Context window grown to {k}K (local model; conversation continues '
                                       'uncompressed)',
                                       '📈 上下文窗口已增至 {k}K（本地模型；对话继续，不做压缩）'],
 'status.nous.rate_limited': ['Your Nous account has hit its rate limit; it resets in {reset}.',
                              '你的 Nous 账号已达速率上限；将于 {reset} 重置。'],
 'status.error.attempts_exhausted': ['❌ {what} {attempts} attempts. The provider may be experiencing issues — try '
                                     'again in a moment.',
                                     '❌ {what} 已尝试 {attempts} 次。提供方可能正在故障 — 请稍后再试。'],
 'status.recovery.stream_interrupted_toolcall': ['↻ Stream interrupted mid tool-call ({tools}) — requesting chunked '
                                                 'retry ({n}/4)...',
                                                 '↻ 工具调用中途流式中断（{tools}）— 正在请求分块重试（{n}/4）…'],
 'status.stream.retry_reconnecting': ['⚠️ {provider} stream {kind} ({error}){suffix} — reconnecting, retry {n}/{m}',
                                      '⚠️ {provider} 流 {kind}（{error}）{suffix} — 正在重连，第 {n}/{m} 次重试'],
 'status.auth.vertex_refreshed_401': ['🔐 Vertex AI token refreshed after 401. Retrying request...',
                                      '🔐 Vertex AI 令牌已在 401 后刷新。正在重试请求…'],
 'status.auth.nous_refreshed_401': ['🔐 Nous agent key refreshed after 401. Retrying request...',
                                    '🔐 Nous agent key 已在 401 后刷新。正在重试请求…'],
 'status.nous.rate_limited_fallback': ['⏳ {message} Trying fallback...', '⏳ {message} 正在尝试兜底…'],
 'status.nous.rate_limited_banner': ['⏳ Your Nous account has hit its rate limit; it resets in {reset}.',
                                     '⏳ 你的 Nous 账号已达速率上限；将于 {reset} 重置。'],
 'status.response.failure_hint_line': ('   ⏱️  {hint}', '   ⏱️  原因：{hint}')}

# Catch-alls, matched last so a more specific line is never swallowed by them.
GENERIC_TEXTS: dict[str, tuple[str, str]] = {'status.error.provider_rejected': ['❌ {label}: {summary}', '❌ {label}：{summary}']}

# Lines whose placeholders hold an expression (e.g. ``{max_retries + 1}``): the regex cannot be
# derived from a template, so it is spelled out here.
EXTRA_RULES: tuple[tuple[str, str], ...] = (('status.error.attempts_exhausted',
  '^❌ (?P<what>.+?) (?P<attempts>\\d+) attempts\\. The provider may be experiencing issues — try again in a '
  'moment\\.$'),
 ('status.response.invalid_api', '^⚠️  Invalid API response \\(attempt (?P<n>\\d+)/(?P<m>\\d+)\\): (?P<details>.+)$'),
 ('status.recovery.stream_interrupted_toolcall',
  '^↻ Stream interrupted mid tool-call \\((?P<tools>.+?)\\) — requesting chunked retry '
  '\\((?P<n>\\d+)/4\\)\\.\\.\\.$'),
 ('status.stream.retry_reconnecting',
  '^⚠️ (?P<provider>\\S+) stream (?P<kind>\\S+) \\((?P<error>.+?)\\)(?P<suffix>.*?) — reconnecting, retry '
  '(?P<n>\\d+)/(?P<m>\\d+)\\.?$'))

# English fragments embedded in a translation, substituted again once the line is rendered.
ZH_SUBSTITUTIONS: tuple[tuple[str, str], ...] = ((' — high-cost request, reduced retry budget', ' — 高成本请求，已减少重试次数'),
 ("The provider's safety filter refused this request", '提供方安全过滤器拒绝了该请求'),
 ("The provider's security certificate could not be verified", '无法验证提供方的安全证书'),
 ("Request still exceeded the provider's size limit after shrinking images", '缩小图片后请求仍超出提供方体积上限'))

# Keys for the routine compaction lines a chat platform keeps quiet.
ROUTINE_KEYS: tuple[str, ...] = ('status.compaction.running',
 'status.compaction.heartbeat',
 'status.compaction.done',
 'status.compaction.pre_api',
 'status.compaction.preflight',
 'status.compaction.idle',
 'status.compaction.too_large',
 'status.compaction.messages',
 'status.compaction.tokens',
 'status.compaction.reduced')

# Cheap pre-filter hints: leading glyphs (derived from the catalog's first characters, non-ASCII
# only) plus the long prefixes of lines that do not start with a glyph. ``translate_status`` runs
# on hot paths such as _vprint, so this rejects the vast majority of ordinary output before the
# hundred-odd regexes are tried. Lines whose first character is a placeholder — the glyph is
# injected by the caller, as in ``{glyph} {provider} — recalled ...`` — are listed here.
_EXTRA_GLYPH_HINTS: tuple[str, ...] = ("🧠",)

GLYPH_HINTS: tuple[str, ...] = tuple(
    sorted({en.lstrip()[0] for en, _zh in {**STATUS_TEXTS, **GENERIC_TEXTS}.values()
            if not en.lstrip()[0].isascii()} | set(_EXTRA_GLYPH_HINTS))
)
PHRASE_HINTS: tuple[str, ...] = tuple(
    sorted({en.lstrip()[:28] for en, _zh in {**STATUS_TEXTS, **GENERIC_TEXTS}.values()
            if en.lstrip()[0].isascii() and en.lstrip()[0].isalpha()})
)

# English diagnostic fragments embedded in a line (provider error text, usually carrying numbers),
# rewritten to Chinese with a regex after rendering.
ZH_SUBSTITUTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"upstream provider timed out \(Cloudflare 524, (\d+)s\)"), r"上游提供方超时（Cloudflare 524，\1 秒）"),
    (re.compile(r"upstream gateway timeout \(504, (\d+)s\)"), r"上游网关超时（504，\1 秒）"),
    (re.compile(r"rate limited by upstream provider \(429\)"), "被上游提供方限流（429）"),
    (re.compile(r"upstream server error \((\d+), (\d+)s\)"), r"上游服务器错误（\1，\2 秒）"),
    (re.compile(r"upstream provider overloaded \((\d+)\)"), r"上游提供方过载（\1）"),
    (re.compile(r"upstream error \(code (\d+), (\d+)s\)"), r"上游错误（代码 \1，\2 秒）"),
    (re.compile(r"fast response \(([\d.]+)s\) — likely rate limited"), r"响应很快（\1 秒）— 很可能被限流"),
    (re.compile(r"slow response \(([\d.]+)s\) — likely upstream timeout"), r"响应很慢（\1 秒）— 很可能是上游超时"),
    (re.compile(r"response time ([\d.]+)s"), r"响应耗时 \1 秒"),
)


_PLACEHOLDER_RE = re.compile(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::[^{}]*)?\}")


def _template_to_regex(template: str) -> str:
    """English source line -> whole-string match regex: literals escaped, ``{name}`` becomes a named group.

    A trailing placeholder uses ``.*`` (it may be empty, e.g. an optional addendum), every other one
    uses ``.+?``. A format spec such as ``{n:,}`` only applies to the human-facing side: what is
    captured is still a string and is filled back as a plain placeholder.
    """
    out, pos = [], 0
    matches = list(_PLACEHOLDER_RE.finditer(template))
    for idx, m in enumerate(matches):
        out.append(re.escape(template[pos:m.start()]))
        remainder = template[m.end():]
        quant = ".*" if (idx == len(matches) - 1 and not remainder) else "[\\s\\S]+?"
        out.append(f"(?P<{m.group('name')}>{quant})")
        pos = m.end()
    out.append(re.escape(template[pos:]))
    return "".join(out)


def _compile_rules() -> tuple[tuple[re.Pattern[str], str], ...]:
    rules = [re.compile(rf"^{_template_to_regex(en)}$") for en, _zh in {**STATUS_TEXTS, **GENERIC_TEXTS}.values()]
    keys = list({**STATUS_TEXTS, **GENERIC_TEXTS}.keys()) + [key for key, _p in EXTRA_RULES]
    return tuple(zip(rules + [re.compile(p) for _k, p in EXTRA_RULES], keys))


_RULES = tuple(sorted(_compile_rules(), key=lambda item: len(item[0].pattern), reverse=True))


def _active_language() -> str:
    from agent.i18n import get_language
    return get_language()


# Which column of a catalog entry belongs to which language. Column 0 is the English original, which
# stays the single source of truth; a language joins the table by adding a column and an entry here.
#
# These translations deliberately live in this module instead of ``locales/<lang>.yaml``: a status
# line is matched *by its English text*, so the two sides belong in one table, and adding keys to the
# locale files would mean translating them into all sixteen shipped languages at once.
_LANG_COLUMN: dict[str, int] = {"zh": 1}


def _catalog_entry(key: str) -> tuple[str, ...] | None:
    return {**STATUS_TEXTS, **GENERIC_TEXTS}.get(key)


def _localized_line(key: str, lang: str) -> str | None:
    """The translation of ``key`` for ``lang``, or None when there is nothing to render.

    Falls back to a ``locales/<lang>.yaml`` entry when the table has no column for that language, so a
    locale that later translates the status keys through the normal catalog also works.
    """
    entry = _catalog_entry(key)
    column = _LANG_COLUMN.get(lang)
    if entry is not None and column is not None and len(entry) > column:
        localized = entry[column]
        return localized or None
    try:
        from agent.i18n import t
        rendered = t(key)
    except Exception:
        return None
    return None if rendered == key else rendered


def _looks_translatable(text: str) -> bool:
    """Cheap pre-filter: only a status-line glyph, CJK text or a known prefix is worth the regexes.

    Serves ``translate_status`` (English input) and ``normalize_status`` (localised input) alike.
    """
    if len(text) > 600:
        return False
    if any(hint in text for hint in GLYPH_HINTS):
        return True
    if any("\u4e00" <= ch <= "\u9fff" for ch in text[:6]):  # an already localised status line
        return True
    stripped = text.strip()
    return any(stripped.startswith(prefix) for prefix in PHRASE_HINTS)


def translate_status(text: str) -> str:
    """Translate one lifecycle/warning status line into the active language.

    Returns the input unchanged when the language is English or the line is not recognised, which
    also makes repeated calls idempotent.
    """
    if not isinstance(text, str) or not text.strip() or not _looks_translatable(text):
        return text
    try:
        if _active_language() == "en":
            return text
    except Exception:
        return text
    for pattern, key in _RULES:
        m = pattern.match(text)
        if not m:
            continue
        translated = _render(key, m.groupdict())
        return text if translated is None else translated
    return text


def _render(key: str, values: dict[str, Any]) -> str | None:
    """Render a translation from the captured values; None (caller keeps the original) when the table
    has no translation for the active language.
    """
    try:
        template = _localized_line(key, _active_language())
    except Exception:
        logger.debug("status i18n render failed for %s", key, exc_info=True)
        return None
    if not template:
        return None
    rendered = _fill(template, values)
    for en_frag, zh_frag in ZH_SUBSTITUTIONS:
        if en_frag in rendered:
            rendered = rendered.replace(en_frag, zh_frag)
    for pattern, repl in ZH_SUBSTITUTION_PATTERNS:
        rendered = pattern.sub(repl, rendered)
    return rendered


def _fill(template: str, values: dict[str, Any]) -> str:
    """Fill captured values back into a ``{placeholder}`` template."""
    return _PLACEHOLDER_RE.sub(lambda m: str(values.get(m.group("name"), "")), template)


@lru_cache(maxsize=8)
def _localized_rules(lang: str) -> tuple[tuple[re.Pattern[str], str], ...]:
    """Reverse rules (localised line -> English original) for checks that still match on English."""
    rules = []
    for key, (en, _columns) in {**STATUS_TEXTS, **GENERIC_TEXTS}.items():
        localized = _localized_line(key, lang)
        if localized and localized != en:
            rules.append((re.compile(rf"^{_template_to_regex(localized)}$"), en))
    return tuple(rules)


def normalize_status(text: str) -> str:
    """Restore a localised status line to its English original for the checks that match on English
    (noise filtering, compaction-progress detection).

    English input or an unrecognised line is returned unchanged, so localising the text never costs
    a Chinese user those classifications.
    """
    if not isinstance(text, str) or not text.strip() or not _looks_translatable(text):
        return text
    try:
        lang = _active_language()
    except Exception:
        return text
    if lang == "en":
        return text
    for pattern, en in _localized_rules(lang):
        m = pattern.match(text)
        if m:
            return _fill(en, m.groupdict())
    return text


__all__ = [
    "STATUS_TEXTS", "GENERIC_TEXTS", "translate_status", "normalize_status",
]
