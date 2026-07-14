# MW · ตาราง [G] → เครื่องมือ §10 → test ID จริง (สัญญา §13 ข้อ 1)

> memory-schema: v1.2 · plan_id: MW · MW-P3-I3 · สร้าง 2026-07-14
> วัตถุประสงค์: ทุกแถว [G] ในตารางแม่ (`mw-spec-draft.md`) ต้องผูกกับ **test ID จริง** ของเครื่องมือ §10 ที่พิสูจน์กลไกได้ · เครื่องตรวจ `scripts/mw-spec-check.py` ยืนยันความครบ
> คอลัมน์ `status`: `mapped` = มี pytest test จริงในชุดของเรา · `external` = เครื่องมือที่มีอยู่แล้วนอกชุด (ตรวจในชุดของมันเอง) · `pending-i2e` = รอ `mw-backend-check` (§10-8) เฟส P4
> รูปแบบบรรทัดตาราง (เครื่องอ่าน): `| <row> | <tool> | <test_id หรือ tag> | <status> |`

## ตารางผูก (32 แถว [G])

| row | tool §10 | test_id / tag | status |
|---|---|---|---|
| I1-01 | §10-8 | test_site_exists_pass | mapped |
| I1-03 | §10-8 | test_broken_file_report_pass | mapped |
| I1-04 | §10-8 | test_es_search_pass | mapped |
| I1-05 | §10-2 | test_pagination_many_items_no_control_fail | mapped |
| I1-07 | §10-7 | ds-check | external |
| I1-10 | §10-2 | test_pagination_many_items_with_control_pass | mapped |
| I1-11 | §10-2 | test_related_self_link_fail | mapped |
| I1-12 | §10-2 | test_file_size_hero_over_cap_fail | mapped |
| I1-13 | §10-2 | test_sticky_cover_manual_not_blocking | mapped |
| I1-14 | §10-2 | test_video_autoplay_no_reduced_motion_fail | mapped |
| I1-16 | §10-1 | test_t4_file_grep_all_present_vs_missing | mapped |
| I1-21 | §10-2 | test_sticky_cover_manual_not_blocking | mapped |
| I1-22 | §10-1 | test_t4_evidence_file_present_absent | mapped |
| I1-23 | §10-6 | hermes-write-permit | external |
| I2-01 | §10-2 | test_good_page_all_pass_deliverable | mapped |
| I2-03 | §10-2 | test_pagination_stale_item_selector_fail | mapped |
| I2-04 | §10-1 | test_fix_b_structural_zero_items_never_closeable | mapped |
| I3-02 | §10-3 | test_reg_mixed_pass_fail_conflict_not_verified | mapped |
| I3-03 | §10-8 | test_siteid_isolation_leak_fail | mapped |
| I3-04 | §10-2 | test_language_bilingual_missing_en_fail | mapped |
| I3-05 | §10-8 | test_data_parity_sampled_label_and_full_field_check | mapped |
| I3-06 | §10-8 | test_form_cycle_pass | mapped |
| I3-07 | §10-8 | test_dashboard_parity_pass | mapped |
| I3-08 | §10-2 | test_good_page_all_pass_deliverable | mapped |
| I3-09 | §10-4 | test_status_lists_active_only | mapped |
| I3-10 | §10-2 | test_language_bilingual_pass | mapped |
| I3-R2 | §10-2 | test_soft_404_phrase_fail | mapped |
| I3-R6 | §10-8 | test_form_cycle_pass | mapped |
| I3-R7 | §10-4 | test_t2_two_clones_one_wins | mapped |
| I4-02 | §10-9 | gitleaks | external |
| I4-03 | §10-Q | Use QA QC | external |
| I5-01 | §10-10 | test_image_expect_contains_pass_and_fail | mapped |

## สรุป
- แถว [G] ทั้งหมด: **32**
- `mapped` (มี pytest test จริงในชุด mw): **28**
- `external` (เครื่องมือมีอยู่แล้ว: ds-check / hermes-write-permit / gitleaks / Use QA QC): **4**
- `pending-i2e` (รอ `mw-backend-check` §10-8 เฟส P4): **0** — I2e เสร็จแล้ว (2026-07-14) ผูก test จริงครบ

**สัญญา §13 ข้อ 1:** ครบ **32/32** แถวผูก test ID/เครื่องมือจริง (100%) · I2e (mw-backend-check) เสร็จ 2026-07-14 ผูก 8 แถว §10-8 กับ test จริง · เครื่อง `mw-spec-check` = COMPLETE
