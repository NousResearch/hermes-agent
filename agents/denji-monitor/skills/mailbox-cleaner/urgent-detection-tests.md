# Urgent Detection Tests — Pattern Matching Scenarios

**Purpose:** Validate the urgent detector's rule-based pattern matching produces correct alert/no-alert decisions.
**How to run:** Feed each scenario to the urgent-detection prompt and verify the output.
**Created:** 19/05/2026

---

## Scenario U1: Interview Invite — Should Alert

| Field | Value |
|-------|-------|
| From | `jane.smith@techrecruiters.com` |
| Subject | "Interview scheduled — Senior PM role at Acme Corp" |
| Body | "We were impressed by your application and would like to schedule an interview..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Subject contains "interview". Sender not in reject list. |
| **Note** | This is the most critical pattern — must alert immediately. |

---

## Scenario U2: Screening Call Request — Should Alert

| Field | Value |
|-------|-------|
| From | `raj.patel@greenhouse.io` |
| Subject | "Quick screening call about your application?" |
| Body | "I'd like to arrange a 15-minute screening call to discuss your experience..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Subject contains "screening". Sender is human pattern (firstname.lastname). |

---

## Scenario U3: Indeed Job Alert — Should NOT Alert

| Field | Value |
|-------|-------|
| From | `alerts@indeed.com` |
| Subject | "5 new Senior PM matches today" |
| Body | "Based on your search criteria, here are 5 new Senior PM roles..." |
| **Expected result** | ❌ NO ALERT |
| **Rule match** | Sender `alerts@indeed.com` is in reject list. |

---

## Scenario U4: Application Confirmation — Should NOT Alert

| Field | Value |
|-------|-------|
| From | `noreply@workable.com` |
| Subject | "Your application has been received" |
| Body | "Thank you for applying to the Senior PM position at Acme Corp." |
| **Expected result** | ❌ NO ALERT |
| **Rule match** | Sender `noreply@workable.com` is in reject list. Subject has no trigger keyword (confirmations are not urgent). |

---

## Scenario U5: Direct Recruiter from Known Folder — Should Alert

| Field | Value |
|-------|-------|
| From | `sarah.jones@hays.com` |
| Subject | "Exciting opportunity at Google" |
| Body | "I came across your profile and think you'd be perfect for a Senior PM role at Google..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Context boost: sender is from a known recruiter agency (Hays). Even though subject doesn't contain a primary trigger word, the recruiter context boost applies. |

---

## Scenario U6: Interview Rescheduled — Should Alert

| Field | Value |
|-------|-------|
| From | `hiring@acmecorp.com` |
| Subject | "Interview rescheduled for Thursday" |
| Body | "Due to a calendar conflict, your interview has been moved to Thursday at 2pm..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Subject contains "interview". Rescheduling is urgent — don't miss the new time. |

---

## Scenario U7: CV-Library Daily Digest — Should NOT Alert

| Field | Value |
|-------|-------|
| From | `daily@cv-library.co.uk` |
| Subject | "Your daily job alerts — 12 new roles" |
| Body | "Here are the latest job matches for your saved search..." |
| **Expected result** | ❌ NO ALERT |
| **Rule match** | Sender `@cv-library.co.uk` is in reject list. |

---

## Scenario U8: Assessment Invite — Should Alert

| Field | Value |
|-------|-------|
| From | `recruitment@techcorp.com` |
| Subject | "You're invited to complete a technical assessment" |
| Body | "As the next step in our process, please complete the technical assessment by Friday..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Subject contains "assessment" + "invited". Sender not in reject list. |

---

## Scenario U9: LinkedIn Connection Message — Should NOT Alert

| Field | Value |
|-------|-------|
| From | `messages-noreply@linkedin.com` |
| Subject | "New message from John on LinkedIn" |
| Body | "Hey Sahil, saw your profile and wanted to connect..." |
| **Expected result** | ❌ NO ALERT |
| **Rule match** | Sender `@linkedin.com` is in reject list. |

---

## Scenario U10: Next Steps Email — Should Alert

| Field | Value |
|-------|-------|
| From | `hiring@startup.io` |
| Subject | "Next steps in your application process" |
| Body | "We were impressed with your interview and would like to discuss next steps..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Subject contains "next step". Sender not in reject list. |

---

## Scenario U11: Technical Test Challenge — Should Alert

| Field | Value |
|-------|-------|
| From | `tech-recruitment@bigco.com` |
| Subject | "Technical challenge for Senior PM position" |
| Body | "Here's your technical challenge. You have 72 hours to complete it..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Subject contains "technical" + "challenge". This is a time-sensitive deliverable. |

---

## Scenario U12: Offer Letter — Should Alert (Highest Priority)

| Field | Value |
|-------|-------|
| From | `hr@dreamcompany.com` |
| Subject | "Pleased to offer you the Senior PM position" |
| Body | "Following your interviews, we are delighted to offer you the position of..." |
| **Expected result** | 🚨 ALERT |
| **Rule match** | Subject contains "pleased to offer". This is the highest-urgency pattern. |

---

## Scenario U13: Multiple Emails in One Run

| # | From | Subject | Expected |
|---|------|---------|----------|
| 1 | `jane@co.com` | "Interview Tuesday" | ALERT |
| 2 | `alerts@indeed.com` | "12 new PM jobs" | NO ALERT |
| 3 | `noreply@lever.co` | "Application received" | NO ALERT |
| 4 | `recruiter@agency.com` | "Next step call?" | ALERT |

**Expected result:** ONE alert listing N=2 matches (emails 1 and 4).

---

## Scenario U14: Same Email Across Two Runs

| Run | From | Subject | Expected |
|-----|------|---------|----------|
| 1 (10:00) | `jane@co.com` | "Interview Tuesday" | ALERT |
| 2 (11:00) | (same email still in inbox) | | NO ALERT — skip already-seen message ID |

**Expected result:** Track by message ID in telemetry log. Don't re-alert.

---

## Scenario U15: No New Emails in Window

| Condition | Expected |
|-----------|----------|
| 0 new emails in sahil_ss@outlook.com in the last 90 minutes | Silent. No Telegram message. Log the run. |

---

## Test Log

| Date | Scenario | Result | Notes |
|------|----------|--------|-------|
| | U1: Interview invite | ☐ | |
| | U2: Screening call | ☐ | |
| | U3: Indeed alert | ☐ | |
| | U4: App confirmation | ☐ | |
| | U5: Recruiter context | ☐ | |
| | U6: Interview reschedule | ☐ | |
| | U7: CV-Library digest | ☐ | |
| | U8: Assessment invite | ☐ | |
| | U9: LinkedIn message | ☐ | |
| | U10: Next steps | ☐ | |
| | U11: Technical test | ☐ | |
| | U12: Offer letter | ☐ | |
| | U13: Multiple in one run | ☐ | |
| | U14: Dedup across runs | ☐ | |
| | U15: Empty window | ☐ | |
