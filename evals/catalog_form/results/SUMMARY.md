# Deferred-catalog form A/B/C, run of 2026-11-09

Tree: origin/main e0f07847b8f3 plus the `listing_servers` knob. 16 tasks x 3 arms; 3 reps for Fable 5.2 and
gpt-5.6-luna, 1 rep for qwen3.8-27b (stopped for time; 3 of its 48 cells are wall timeouts). Regenerate with
`run.py --report`. Per-cell JSON and transcripts are not checked in.

```
cells: 336   billed (sum of per-cell dollars): $7.46

## anthropic/claude-fable-5.2  ($6.77)

| arm | deferred+continuation: task done | right tool reached | wrong-tool calls | model calls to first correct use | controls done | distractors done | false-discovery bridge calls | noise/error cells | tokens per cell (all-in) | fresh+write input per cell | $ per cell | $ per completed task |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 36/36 | 36/36 | 0 | 1.86 | 6/6 | 6/6 | 0 | 0 | 28,512 | 1,701 | 0.0384 | 0.0384 |
| names | 36/36 | 36/36 | 0 | 1.92 | 6/6 | 6/6 | 0 | 0 | 22,542 | 1,745 | 0.0368 | 0.0368 |
| summary | 33/36 | 33/36 | 0 | 3.33 | 6/6 | 6/6 | 0 | 0 | 32,189 | 3,257 | 0.0658 | 0.0702 |

| task | kind | full done | names done | summary done | full reached | names reached | summary reached | full wrong | names wrong | summary wrong | full calls | names calls | summary calls | full first-correct | names first-correct | summary first-correct | full tok | names tok | summary tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bl_cli_datablocks | deferred | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 0 | 0 | 0 | 3.0 | 3.0 | 7.3 | 2.0 | 2.0 | 7.0 | 28,344 | 23,029 | 65,580 |
| bl_live_missing | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 2.3 | 3.0 | 3.0 | 1.3 | 2.0 | 2.0 | 21,912 | 22,456 | 19,945 |
| bl_cli_linked | deferred | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 1/3 | 0 | 0 | 0 | 3.0 | 3.0 | 6.3 | 2.0 | 2.0 | 6.0 | 28,484 | 23,278 | 65,724 |
| bl_object_modifiers | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.0 | 3.0 | 7.0 | 2.0 | 2.0 | 6.0 | 27,742 | 22,474 | 59,050 |
| meshy_printability | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.0 | 3.0 | 4.0 | 2.0 | 2.0 | 3.0 | 28,083 | 22,774 | 28,219 |
| meshy_balance | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 2.0 | 2.0 | 3.0 | 1.0 | 1.0 | 2.0 | 18,146 | 14,595 | 18,811 |
| meshy_remesh | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.0 | 3.0 | 4.0 | 2.0 | 2.0 | 3.0 | 27,881 | 22,579 | 26,725 |
| el_stems | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 5.0 | 4.3 | 5.0 | 2.0 | 2.0 | 3.0 | 51,349 | 36,408 | 40,235 |
| el_draft_plan | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.0 | 3.0 | 4.0 | 2.0 | 2.0 | 3.0 | 28,176 | 22,957 | 27,993 |
| ns_threads | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.0 | 3.0 | 4.0 | 2.0 | 2.0 | 3.0 | 27,732 | 22,434 | 27,212 |
| core_sfx | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 5.0 | 4.7 | 5.0 | 2.0 | 2.0 | 2.0 | 50,682 | 37,724 | 35,886 |
| meshy_underspecified | continuation | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.0 | 3.0 | 4.0 | 2.0 | 2.0 | 3.0 | 42,862 | 29,365 | 48,689 |
| ctl_count_refunds | control | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 18,523 | 15,005 | 12,650 |
| ctl_write_todo | control | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 18,602 | 15,054 | 12,691 |
| dis_models_csv | distractor | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 18,342 | 14,895 | 12,486 |
| dis_blender_script | distractor | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 19,329 | 15,647 | 13,122 |

## openai/gpt-5.6-luna  ($0.38)

| arm | deferred+continuation: task done | right tool reached | wrong-tool calls | model calls to first correct use | controls done | distractors done | false-discovery bridge calls | noise/error cells | tokens per cell (all-in) | fresh+write input per cell | $ per cell | $ per completed task |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 36/36 | 36/36 | 0 | 2.78 | 6/6 | 6/6 | 0 | 0 | 22,892 | 7,161 | 0.0024 | 0.0024 |
| names | 34/36 | 35/36 | 0 | 3.06 | 6/6 | 6/6 | 0 | 0 | 23,742 | 6,365 | 0.0023 | 0.0024 |
| summary | 27/36 | 27/36 | 13 | 4.67 | 6/6 | 6/6 | 0 | 0 | 38,336 | 6,982 | 0.0033 | 0.0040 |

| task | kind | full done | names done | summary done | full reached | names reached | summary reached | full wrong | names wrong | summary wrong | full calls | names calls | summary calls | full first-correct | names first-correct | summary first-correct | full tok | names tok | summary tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bl_cli_datablocks | deferred | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 0 | 0 | 0 | 3.3 | 3.3 | 6.3 | 2.3 | 2.3 | 4.0 | 22,233 | 18,458 | 44,370 |
| bl_live_missing | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.0 | 3.3 | 4.0 | 2.0 | 2.3 | 3.0 | 19,483 | 18,076 | 18,637 |
| bl_cli_linked | deferred | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 1/3 | 0 | 0 | 0 | 3.3 | 4.0 | 14.0 | 2.3 | 3.0 | 13.0 | 22,326 | 22,594 | 118,332 |
| bl_object_modifiers | deferred | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 0 | 0 | 6 | 3.3 | 4.0 | 15.3 | 2.3 | 3.0 | 11.5 | 21,887 | 22,496 | 114,672 |
| meshy_printability | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 4.0 | 4.0 | 5.0 | 3.0 | 3.0 | 4.0 | 26,755 | 21,993 | 29,356 |
| meshy_balance | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 3.3 | 4.0 | 4.3 | 2.3 | 3.0 | 3.3 | 21,546 | 21,477 | 20,724 |
| meshy_remesh | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 4.0 | 5.3 | 6.0 | 3.0 | 3.0 | 4.0 | 26,499 | 30,536 | 31,672 |
| el_stems | deferred | 3/3 | 2/3 | 1/3 | 3/3 | 2/3 | 1/3 | 0 | 0 | 0 | 4.7 | 7.3 | 6.3 | 3.0 | 3.0 | 4.0 | 31,613 | 63,351 | 49,974 |
| el_draft_plan | deferred | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | 0 | 0 | 0 | 4.3 | 4.7 | 1.7 | 3.3 | 3.7 | - | 28,991 | 26,438 | 7,962 |
| ns_threads | deferred | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 4.0 | 4.0 | 5.0 | 3.0 | 3.0 | 4.0 | 26,206 | 21,654 | 25,514 |
| core_sfx | deferred | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 5.0 | 6.0 | 6.3 | 3.7 | 3.7 | 3.3 | 33,743 | 35,666 | 32,097 |
| meshy_underspecified | continuation | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 7 | 4.0 | 4.7 | 10.0 | 3.0 | 3.7 | 4.3 | 32,682 | 33,928 | 81,534 |
| ctl_count_refunds | control | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 13,074 | 10,832 | 9,252 |
| ctl_write_todo | control | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.3 | - | - | - | 13,073 | 10,795 | 10,798 |
| dis_models_csv | distractor | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 12,988 | 10,694 | 9,148 |
| dis_blender_script | distractor | 3/3 | 3/3 | 3/3 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 13,181 | 10,880 | 9,327 |

## qwen/qwen3.8-27b  ($0.30)

| arm | deferred+continuation: task done | right tool reached | wrong-tool calls | model calls to first correct use | controls done | distractors done | false-discovery bridge calls | noise/error cells | tokens per cell (all-in) | fresh+write input per cell | $ per cell | $ per completed task |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 9/12 | 9/12 | 1 | 2.78 | 2/2 | 2/2 | 0 | 2 | 40,532 | 9,851 | 0.0051 | 0.0063 |
| names | 10/12 | 10/12 | 5 | 4.00 | 2/2 | 2/2 | 0 | 0 | 57,235 | 12,600 | 0.0078 | 0.0089 |
| summary | 10/12 | 10/12 | 8 | 5.50 | 2/2 | 2/2 | 0 | 1 | 61,131 | 11,258 | 0.0062 | 0.0071 |

| task | kind | full done | names done | summary done | full reached | names reached | summary reached | full wrong | names wrong | summary wrong | full calls | names calls | summary calls | full first-correct | names first-correct | summary first-correct | full tok | names tok | summary tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bl_cli_datablocks | deferred | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 0 | 0 | 0 | 13.0 | 9.0 | 4.0 | - | 8.0 | 3.0 | 190,836 | 84,989 | 27,265 |
| bl_live_missing | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 0 | 0 | 4.0 | 3.0 | 4.0 | 3.0 | 2.0 | 2.0 | 30,527 | 19,825 | 25,469 |
| bl_cli_linked | deferred | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0 | 0 | 0 | - | 12.0 | - | - | - | - | - | 152,709 | - |
| bl_object_modifiers | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 0 | 0 | 2.0 | 4.0 | 9.0 | 1.0 | 3.0 | 8.0 | 14,888 | 26,392 | 68,582 |
| meshy_printability | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 0 | 4 | 4.0 | 4.0 | 19.0 | 3.0 | 3.0 | 18.0 | 32,000 | 27,194 | 257,863 |
| meshy_balance | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 0 | 0 | 3.0 | 5.0 | 4.0 | 2.0 | 4.0 | 3.0 | 22,713 | 31,239 | 23,048 |
| meshy_remesh | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 1 | 0 | 5.0 | 8.0 | 5.0 | 4.0 | 7.0 | 4.0 | 40,369 | 62,104 | 32,744 |
| el_stems | deferred | 1/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1 | 0 | 0 | 6.0 | 11.0 | 11.0 | 3.0 | - | - | 49,835 | 118,880 | 98,583 |
| el_draft_plan | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 0 | 0 | 3.0 | 4.0 | 5.0 | 2.0 | 3.0 | 4.0 | 24,297 | 28,185 | 33,962 |
| ns_threads | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 0 | 0 | 4.0 | 4.0 | 4.0 | 3.0 | 3.0 | 3.0 | 30,274 | 25,636 | 24,465 |
| core_sfx | deferred | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0 | 0 | 0 | 6.0 | 6.0 | 6.0 | 4.0 | 3.0 | 4.0 | 47,518 | 40,346 | 35,541 |
| meshy_underspecified | continuation | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 0 | 4 | 4 | - | 21.0 | 21.0 | - | 4.0 | 6.0 | - | 247,734 | 232,948 |
| ctl_count_refunds | control | 1/1 | 1/1 | 1/1 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 4.0 | 2.0 | 3.0 | - | - | - | 31,183 | 13,070 | 16,703 |
| ctl_write_todo | control | 1/1 | 1/1 | 1/1 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 3.0 | 2.0 | 2.0 | - | - | - | 22,780 | 12,516 | 11,072 |
| dis_models_csv | distractor | 1/1 | 1/1 | 1/1 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 2.0 | - | - | - | 14,780 | 12,484 | 11,046 |
| dis_blender_script | distractor | 1/1 | 1/1 | 1/1 | bridge 0 | bridge 0 | bridge 0 | 0 | 0 | 0 | 2.0 | 2.0 | 3.0 | - | - | - | 15,441 | 12,460 | 17,677 |
```
