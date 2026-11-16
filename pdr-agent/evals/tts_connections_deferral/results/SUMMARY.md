# Results: defer text_to_speech and manage_connections (2026-11-09)

Arms: `base` = origin/main e0f07847b8f3 (both tools eager); `pr` = same tree with both tools in
`tools.tool_search.defer`. 13 tasks x 3 reps x 2 arms x 2 models = 156 cells, none errored, none unscored.
Toolsets: clarify, code_execution, cronjob, file, memory, session_search, skills, terminal, todo, vision,
image_gen, tts, connections (no browser, web or delegation: a narrower prefix than a default session).

## Verdict

- `text_to_speech` holds: reached 12/12 on both arms on both models, arguments valid on the first call
  every time, no false trigger on the negative. Deferred by default in this change.
- `manage_connections` does not hold on the weak model: "Install the Linear MCP server" went 3/3 -> 0/3
  on qwen3.8-27b. It never searched; it used the terminal (`hermes mcp install linear`) instead. Kept eager.
- At ceiling on Fable 5.2 (39/39 both arms): that is no-regression on these flows, not proof of safety.

## Per model

| model | arm | completed | tokens per completed task | $ per completed task |
|---|---|---|---|---|
| claude-fable-5.2 | base | 39/39 | 36,457 | 0.0485 |
| claude-fable-5.2 | pr | 39/39 | 39,742 | 0.0577 |
| qwen3.8-27b | base | 38/39 | 42,611 | 0.0052 |
| qwen3.8-27b | pr | 33/39 | 94,522 | 0.0080 |

## Per tool group (reached = stub called; n = cells)

| model | group | base reached | pr reached | base 1st call valid | pr 1st call valid | base calls | pr calls | base tok/cell | pr tok/cell |
|---|---|---|---|---|---|---|---|---|---|
| fable-5.2 | tts, should use | 12/12 | 12/12 | 12/12 | 12/12 | 3.5 | 4.0 | 46,671 | 46,822 |
| fable-5.2 | tts negative | 0/3 | 0/3 | - | - | 2.0 | 2.0 | 25,476 | 21,278 |
| fable-5.2 | connections, should use | 15/15 | 15/15 | 15/15 | 15/15 | 3.1 | 4.3 | 39,575 | 50,939 |
| fable-5.2 | connections negative | 0/3 | 0/3 | - | - | 1.0 | 1.0 | 12,702 | 10,592 |
| fable-5.2 | controls | - | - | - | - | 2.0 | 2.0 | 25,602 | 21,400 |
| qwen3.8-27b | tts, should use | 12/12 | 12/12 | 12/12 | 12/12 | 4.9 | 6.8 | 59,825 | 71,855 |
| qwen3.8-27b | tts negative | 0/3 | 0/3 | - | - | 2.0 | 2.0 | 22,198 | 17,053 |
| qwen3.8-27b | connections, should use | 14/15 | 12/15 | 14/15 | 12/15 | 3.9 | 7.5 | 41,637 | 136,969 |
| qwen3.8-27b | connections negative | 0/3 | 0/3 | - | - | 1.0 | 1.0 | 10,058 | 8,975 |
| qwen3.8-27b | controls | - | - | - | - | 2.7 | 2.3 | 29,999 | 20,728 |

## Per task (score, x/3)

| task | fable base | fable pr | qwen base | qwen pr | note |
|---|---|---|---|---|---|
| tts_read_aloud | 3 | 3 | 3 | 3 | |
| tts_voice_note | 3 | 3 | 3 | 3 | |
| tts_indirect_driving | 3 | 3 | 3 | 3 | |
| tts_indirect_mp3 | 3 | 3 | 3 | 3 | |
| tts_negative | 3 | 3 | 3 | 3 | no false trigger on either arm |
| conn_connect_gmail | 3 | 3 | 3 | 3 | |
| conn_status_notion | 3 | 3 | 2 | 1.5 | qwen pr reached the tool 3/3 (base 2/3); it lost half points for also connecting after the scripted "go ahead" |
| conn_required_recovery | 3 | 3 | 3 | 3 | CONNECTION_REQUIRED -> manage_connections connect -> retry, through the bridge |
| conn_mcp_install | 3 | 3 | 3 | **0** | qwen pr never called tool_search; terminal + `hermes mcp install` instead, 12 calls and 437k tokens per cell |
| conn_second_account | 3 | 3 | 3 | 3 | |
| conn_negative | 3 | 3 | 3 | 3 | |
| ctl_write_file | 3 | 3 | 3 | 3 | |
| ctl_count_lines | 3 | 3 | 3 | 3 | |

## Findings

1. Prefix saved per request, real Fable 5.2 token counts: text_to_speech 577, manage_connections 1,563,
   minus 20 and 18 for their catalog lines. This change (text_to_speech only) saves 557 per request;
   1,245 - 20 with the ElevenLabs schema.
2. A session that uses a deferred tool pays one extra model call (`tool_describe`): Fable +1 call on the
   explicit TTS tasks, about +7,700 tokens per cell with both tools deferred, mostly cache reads.
3. Break-even for text_to_speech: 557 tokens saved on every call of every session against at most
   7,700 extra in a session that speaks. A speaking session breaks even after 14 model calls; across
   sessions of 10 calls deferral wins unless more than 7 in 10 sessions use the tool.
4. The weak model's MCP-install miss is the affordance-collapse class: with the schema invisible the
   catalog line ("Connect the user to apps.") says nothing about MCP servers, and the model takes the
   terminal route the schema forbids. A catalog line that names MCP install is the experiment to run
   before deferring manage_connections.
5. Harness corrections, applied to both arms: conn_negative and conn_second_account were rerun (24 cells
   discarded) after a scripted reply fired on a connection string ending in `?sslmode=` and the stub's
   status listing did not show a newly added account.


## Follow-up: can manage_connections be deferred with a better catalog line? (three models, four arms)

`battery.py`, one tree, 44 cells per arm per model (4 MCP-install phrasings x 5, 8 other tasks x 3), 528 cells,
1 wall timeout (scored 0), 0 unscored. Arms: A eager, B deferred, C deferred with the first sentence reworded to
"Connect the user's app accounts and install MCP servers.", D eager with that sentence.

| model | arm | done | wrong-route cells | model calls/task | tokens/task | tokens/completed task | $/task |
|---|---|---|---|---|---|---|---|
| claude-fable-5.2 | A | 44/44 | 3 | 2.55 | 30,941 | 30,941 | 0.0178 |
| claude-fable-5.2 | B | 44/44 | 18 | 4.84 | 62,022 | 62,022 | 0.0694 |
| claude-fable-5.2 | C | 44/44 | 8 | 3.73 | 44,029 | 44,029 | 0.0371 |
| claude-fable-5.2 | D | 44/44 | 2 | 2.61 | 31,800 | 31,800 | 0.0240 |
| gpt-5.6-luna | A | 44/44 | 0 | 2.25 | 17,180 | 17,180 | 0.0009 |
| gpt-5.6-luna | B | 44/44 | 14 | 6.23 | 61,086 | 61,086 | 0.0029 |
| gpt-5.6-luna | C | 44/44 | 3 | 4.41 | 32,922 | 32,922 | 0.0015 |
| gpt-5.6-luna | D | 44/44 | 0 | 2.30 | 17,607 | 17,607 | 0.0009 |
| qwen3.8-27b | A | 39/44 | 10 | 6.25 | 105,147 | 118,627 | 0.0097 |
| qwen3.8-27b | B | 32/44 | 17 | 10.43 | 226,970 | 312,083 | 0.0216 |
| qwen3.8-27b | C | 39/44 | 16 | 8.41 | 132,233 | 149,186 | 0.0110 |
| qwen3.8-27b | D | 42/44 | 12 | 7.18 | 134,450 | 140,852 | 0.0129 |

Pass bar (against A, on every model: no fewer done, no more wrong-route cells, calls/task up by at most 0.2,
tokens/task not higher): B fails on all three models, C fails on all three. The reworded line roughly halves the
cost of deferral (B to C) and restores tasks done on qwen, but every task that uses the tool still pays one to
three extra model calls for describe and call, so deferral only wins in sessions that never touch connections
(1,536 Fable tokens saved per request). On this per-task bar manage_connections would stay eager; #153254 ships
arm C anyway, because opted-in telemetry shows about 3.6% of reporting installs calling it on a given day.

Limit found in review: `mcp_setup_github` asks for a server the bundled MCP catalog does not carry. The stub
checks argument shape only, so those cells (5 per arm, the same on every arm) score an install the real tool
would refuse.
