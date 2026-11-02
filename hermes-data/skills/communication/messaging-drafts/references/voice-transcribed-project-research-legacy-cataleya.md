# Worked Example: Voice-Transcribed Project Name Resolution — Legacy Cataleya

**Date:** 2026-07-29  
**Trigger:** User said "Century Katalya on Kaningam Road next to Ranka Chambers" via voice. Requested all project details for sharing with a colleague.  
**Reference:** P2.3 in SKILL.md

## What the User Said (STT Output)

> "Century Katalya on Kaningam Road next to Ranka Chambers"

## The Actual Project

**Legacy Cataleya** by **Legacy Global Projects Pvt. Ltd.**  
No. 30, Cunningham Road, Vasanth Nagar, Bengaluru — adjacent to Ranka Chambers (31, Cunningham Road).

Both "Century" (→ Legacy) and "Katalya" (→ Cataleya) were garbled by voice transcription. The road "Kaningam" → Cunningham.

## Search Sequence

1. **Direct search** for "Century Katalya" → zero results across all web search engines.
2. **Phonetic variants:** "Century Cataleya", "Century Catalia", "Century Katalia" → zero results.
3. **Landmark search:** Searched Cunningham Road + Ranka Chambers for Century Real Estate projects. Found no Century projects there.
4. **Real estate portal search:** Searched "Cunningham Road residential project" on 99acres, MagicBricks, Housing.com. Found "Legacy Cataleya" on Cunningham Road.
5. **Clarify with user:** Presented options including Legacy Cataleya, Century Regalia, Century Renata. User confirmed "Legacy Cataleya on Cunningham Road."

## Research Sources

### Primary: RERA Aggregator
- **aurumproptech.in** — returned full RERA record with approval dates, financial snapshot, milestones, architect details, FAR, land area, and status.

### Secondary: Official Site
- **legacy.in/cataleya** — confirmed unit sizes (4,800–5,100 sq.ft.), 24 units, G+14 floors, amenities list.

### Tertiary: Real Estate Portals
- MagicBricks, Roof&Floor, Housing.com, PropHunt — address confirmation, pricing, possession timelines.

## Key Data Collected

| Detail | Value |
|---|---|
| **RERA No.** | PRM/KA/RERA/1251/309/PR/171016/000867 |
| **Address** | No. 30, Cunningham Road, Vasanth Nagar, Bengaluru - 560052 |
| **Builder** | Legacy Global Projects Pvt. Ltd. |
| **Land Area** | 3,035 sqm (~0.76 acres) |
| **FAR** | 3.53 |
| **Floors** | G+14 |
| **Units** | 24 (4 BHK, 4,800–5,100 sq.ft.) |
| **Approving Authority** | BBMP |
| **RERA Approval** | 15 Oct 2017 |
| **Extended Completion** | 30 Sep 2026 |
| **Status** | Ongoing (54% complete) |
| **Architect** | Prasad Consultants (CA/2001/27523) |

## Pattern Takeaways

1. **Every word can be wrong.** Don't assume only one term is garbled — the builder name, project name, and road name can all be corrupted independently.
2. **Landmarks are the most reliable anchor.** "Next to Ranka Chambers" survived the voice garbling and was the most useful clue.
3. **Clarify with specific options, not "what did you mean."** Presenting 3–4 concrete candidates from real search results gets a faster, more accurate response.
4. **RERA aggregator sites** (aurumproptech.in, propzilla.in) provide richer structured data than the official RERA portal which times out on direct scraping.
5. **Always note the correction.** The user needs to know what name the recipient will actually see in the message vs what they dictated.
