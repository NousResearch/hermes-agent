# LCM Arm-B — Live Node-Served Long-Session Recovery

Model: claude-haiku-4-5 · Profile: aegis · N: 40
Gate-eligible (N>=180): False
**Verdict: FAIL**

## Gate summary
- Condensation fired (>=1 depth-1 node): 40/40
- Fact preserved in a depth>=1 node: 40/40
- Node-served recall: 35/40 (0.875)
- Wilson 95% lower bound: 0.7389 (required >= 0.90)
- Confident-wrong: 4 (required 0)
- Duration: 6772.5s

## Trial records
| idx | session | node | sentinel_in_node | correct | leaves | condensed |
|---|---|---|---|---|---|---|
| 0 | 20260618_131624_767f3b | 928 | True | True | 0 | 1 |
| 1 | 20260618_131624_767f3b | 928 | True | True | 0 | 1 |
| 2 | 20260618_132132_d39148 | 933 | True | True | 0 | 1 |
| 3 | 20260618_132132_d39148 | 933 | True | True | 0 | 1 |
| 4 | 20260618_083719_aa7a91 | 668 | True | False | 0 | 1 |
| 5 | 20260618_083719_aa7a91 | 668 | True | False | 0 | 1 |
| 6 | 20260618_133225_c89469 | 943 | True | True | 0 | 1 |
| 7 | 20260618_133225_c89469 | 943 | True | True | 0 | 1 |
| 8 | 20260618_133754_6fa8bf | 948 | True | True | 0 | 1 |
| 9 | 20260618_133754_6fa8bf | 948 | True | True | 0 | 1 |
| 10 | 20260618_134329_834dcc | 953 | True | True | 0 | 1 |
| 11 | 20260618_134329_834dcc | 953 | True | True | 0 | 1 |
| 12 | 20260618_134917_24135b | 958 | True | True | 0 | 1 |
| 13 | 20260618_134917_24135b | 958 | True | True | 0 | 1 |
| 14 | 20260618_135458_9c4490 | 963 | True | True | 0 | 1 |
| 15 | 20260618_135458_9c4490 | 963 | True | True | 0 | 1 |
| 16 | 20260618_140131_c0f38d | 968 | True | True | 0 | 1 |
| 17 | 20260618_140131_c0f38d | 968 | True | True | 0 | 1 |
| 18 | 20260618_140802_c468f0 | 974 | True | True | 0 | 1 |
| 19 | 20260618_140802_c468f0 | 974 | True | True | 0 | 1 |
| 20 | 20260618_141445_b6a94c | 980 | True | True | 0 | 1 |
| 21 | 20260618_141445_b6a94c | 980 | True | True | 0 | 1 |
| 22 | 20260618_142004_5a9f5c | 986 | True | True | 0 | 1 |
| 23 | 20260618_142004_5a9f5c | 986 | True | True | 0 | 1 |
| 24 | 20260618_064920_b381ef | 563 | True | False | 0 | 1 |
| 25 | 20260618_142512_4b1d51 | 991 | True | True | 0 | 1 |
| 26 | 20260618_142512_4b1d51 | 991 | True | False | 0 | 1 |
| 27 | 20260618_142512_4b1d51 | 991 | True | False | 0 | 1 |
| 28 | 20260618_143712_b1a04c | 1002 | True | True | 0 | 1 |
| 29 | 20260618_143712_b1a04c | 1002 | True | True | 0 | 1 |
| 30 | 20260618_144245_3d14c8 | 1007 | True | True | 0 | 1 |
| 31 | 20260618_144245_3d14c8 | 1007 | True | True | 0 | 1 |
| 32 | 20260618_144805_65e1c1 | 1012 | True | True | 0 | 1 |
| 33 | 20260618_144805_65e1c1 | 1012 | True | True | 0 | 1 |
| 34 | 20260618_145320_0d73b2 | 1017 | True | True | 0 | 1 |
| 35 | 20260618_145320_0d73b2 | 1017 | True | True | 0 | 1 |
| 36 | 20260618_145853_4afdb9 | 1022 | True | True | 0 | 1 |
| 37 | 20260618_145853_4afdb9 | 1022 | True | True | 0 | 1 |
| 38 | 20260618_150356_5df150 | 1027 | True | True | 0 | 1 |
| 39 | 20260618_150356_5df150 | 1027 | True | True | 0 | 1 |
