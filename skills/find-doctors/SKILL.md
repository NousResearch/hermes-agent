---
name: find-doctors
slug: find-doctors
version: 0.1.0
description: Find the strongest mainland China doctors and hospitals for a specific disease using evidence-based medical source triangulation. Use when a patient or family member asks “这个病找哪个医生/医院最好”, “中国大陆哪里看这个病最强”, “帮我按病种找专家”, “想挂名医/会诊/二诊”, or needs a ranked shortlist of doctors, hospitals, departments, and appointment paths.
tags: [find-doctors, find-dcotors, doctors, hospitals, China, mainland China, 医生, 医院, 找医生, 名医, 专家, 挂号, 二诊]
license: MIT
metadata:
  openclaw:
    emoji: "🏥"
    requires:
      bins: []
    os: ["linux", "darwin", "win32"]
  hermes:
    tags: [find-doctors, find-dcotors, 医生推荐, 医院推荐, 中国大陆, 就医导航]
related_skills: [health-checkup-report]
---

# Find Doctors

Use this skill to help patients identify the best-fit mainland China doctors and hospitals for a named disease, suspected disease, or treatment need.

Treat `find-dcotors`, `找医生`, `找专家`, and `医院推荐` as aliases.

This skill is a medical navigation and research workflow. It does not diagnose, prescribe, guarantee outcomes, or replace an in-person clinician.

## Required Research Posture

Because hospital rankings, doctor rosters, outpatient schedules, platform reviews, and official pages change frequently, use live search or browser tools whenever available.

If live search is unavailable, say that the ranking is provisional and ask the user to enable browsing or provide sources/screenshots.

Read `references/methodology.md` before producing a ranked doctor shortlist.

## Intake

Collect only details that materially change the search:

- disease name, suspected disease, or procedure/treatment needed
- confirmed diagnosis vs symptoms only
- adult, pediatric, pregnancy, rare disease, oncology stage, emergency status, or recurrent/refractory case if relevant
- current city/province and whether the patient can travel to Beijing, Shanghai, Guangzhou, Chengdu, Wuhan, Hangzhou, Nanjing, Xi'an, etc.
- preference for public tertiary hospital, international department, specialist outpatient, multidisciplinary clinic, second opinion, surgery, medication, radiotherapy, interventional treatment, rehab, or chronic follow-up
- insurance/self-pay constraints and urgency

If the user only gives symptoms, first help them choose the likely first department and ask them to seek a proper diagnosis. Do not rank disease specialists as if the diagnosis is confirmed.

## Safety First

Start with urgent escalation if the user mentions red flags such as:

- chest pain, stroke-like symptoms, severe breathing difficulty, fainting, confusion
- severe bleeding, vomiting blood, black stool, major trauma, severe abdominal pain
- infant/child high fever with lethargy, seizure, dehydration, or breathing difficulty
- cancer treatment complications, severe infection signs, post-operative acute symptoms
- suicidal thoughts or acute psychiatric danger

For red flags, advise immediate emergency care or local urgent evaluation before specialist ranking.

## Evidence Workflow

1. Normalize the disease.
   - Identify the Chinese disease name, common synonyms, subspecialty, and first-choice department.
   - Separate diagnosis, staging, and treatment mode. Example: `肺癌` is not enough; `早期肺结节手术`, `EGFR阳性晚期肺癌靶向治疗`, and `放疗` point to different teams.

2. Build a hospital shortlist.
   - Use disease-relevant specialty rankings, not overall hospital fame.
   - Cross-check at least two classes of evidence when possible:
     - Fudan hospital specialty reputation / specialty comprehensive rankings
     - Chinese Academy of Medical Sciences STEM/ASTEM or discipline-level hospital science metrics
     - National clinical key specialties or national/provincial medical centers
     - disease-specific centers, MDTs, national quality-control centers, registries, or official hospital specialty pages
   - Prefer public tertiary hospitals for complex cases unless the user explicitly wants private care or convenience.

3. Discover doctor candidates.
   - Start from official hospital department/team pages and official appointment pages.
   - Add doctors from guideline/consensus authorship, academic society roles, clinical trial principal investigators, specialty committee roles, and major disease-center leadership.
   - Use patient platforms such as 好大夫在线, 微医, 名医汇, 京东健康, or hospital mini-programs as secondary evidence for accessibility, case volume signals, patient language, and recent appointment availability.
   - Verify doctor identity and practice location when possible with official hospital pages or national physician registration query.

4. Score fit, not celebrity.
   - Disease/procedure fit and case experience: 40
   - Hospital/department strength: 25
   - Individual doctor evidence: 20
   - Accessibility and logistics: 10
   - Source reliability and recency: 5
   - Penalize sponsored rankings, miracle claims, unverifiable titles, stale pages, and doctors whose expertise does not match the exact disease subtype.

5. Output a useful shortlist.
   - Provide `全国优先`, `区域优先`, and `更容易挂到/先看` tiers when appropriate.
   - For each candidate include doctor, hospital, department, why this match is strong, evidence used, appointment path, and caveats.
   - Keep the tone practical: `我找到的证据支持优先考虑`, not `绝对最好`.

## Output Contract

Use Chinese by default unless the user asks otherwise.

For a full answer:

```text
一句话建议：
<best practical path>

先确认：
- <diagnosis/stage/treatment details that would change ranking>

全国优先候选：
1. <医生>｜<医院>｜<科室>
   为什么：<disease-specific reason>
   证据：<ranking/official page/guideline/society/platform signals>
   怎么挂：<official appointment path or platform>
   注意：<availability/logistics/caveat>

区域/可及性候选：
...

选择策略：
- <when to choose top national center>
- <when to choose local tertiary hospital first>
- <when second opinion or MDT is worth it>

来源与不确定性：
- <list source types and dates>
- <state missing info or stale pages>
```

For a quick answer, return the top 3 hospitals/doctor directions and one next step.

## Boundaries

- Do not claim that a doctor is the “best” without explaining the evidence and uncertainty.
- Do not use a single platform score, ad list, or comment count as the ranking basis.
- Do not recommend medicine, dose, stopping treatment, or bypassing emergency care.
- Do not expose personal identifiers. Ask users to redact names, ID numbers, phone numbers, medical record numbers, and exact addresses from uploaded records.
- Do not fabricate appointment availability, pricing, surgery volume, titles, or hospital affiliations.
