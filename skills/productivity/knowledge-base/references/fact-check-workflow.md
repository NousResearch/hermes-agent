# Fact-Check Workflow for Health & Science Claims

When a content creator makes specific claims (especially medical/health), follow this
verification sequence before assigning a confidence level. Depth scales with claim severity.

## Step 1: Verify Creator Credentials

```
web_search(query="<creator name> <specialty> credentials qualifications")
```

Checklist:
- Does the claimed title ("Dr", "MD", "PhD") match verifiable credentials?
- Is the creator practicing in the field they're claiming expertise in?
- Does their platform presence (Instagram bio, website, LinkedIn) match search results?
- For "Dr" without specifics: search for the exact Instagram handle, full name, and specialty

**Red flags:**
- "Dr" title with no specialty listed publicly
- Credentials from unrelated fields (e.g., PhD in economics making neurosurgery claims)
- Creator found in search results but with different specialty than content suggests

## Step 2: Search for Cited Studies

When the creator says "a 2024 study showed...":
```
web_search(query="<exact study description> <year> <journal keywords>")
```

Try multiple phrasings:
- Exact phrase from the video
- Broader topic + year + "study" / "clinical trial"
- Topic + "PubMed" / "RCT" / "systematic review"

**If study is found:** evaluate sample size, methodology, journal quality, replication status.
**If NOT found:** note it explicitly in the note. A missing cited study is a significant
credibility signal.

## Step 3: Evaluate Physiological Plausibility

For health claims that invoke a mechanism ("the oil pulls waste from your aqueous humor"):

1. Identify the anatomical/physiological pathway claimed
2. Check if that pathway exists in medical literature
3. Check if the intervention can physically reach the claimed target

Example (Netra Tarpanam):
- Claim: oil on eye surface → drains waste from aqueous humor → tissue regeneration
- Reality: aqueous humor is behind the cornea (impermeable barrier). Lymphatic drainage
  of the eye exists but via uveoscleral pathway and conjunctival lymphatics — NOT through
  the corneal surface.
- Conclusion: mechanism not physiologically plausible as described

## Step 4: Search for Opposing Views

```
web_search(query="<claim> debunked skepticism criticism")
web_search(query="<treatment> risks dangers contraindications")
```

What do conventional medical sources say? Are there documented risks?

## Step 5: Assess Risk Level

For health claims, explicitly assess:
- **What happens if a patient believes this instead of conventional treatment?**
  - Cataract example: delaying surgery → permanent vision loss
  - Glaucoma example: stopping pressure-lowering drops → irreversible optic nerve damage
- **Is the claim about managing symptoms or curing a disease?**
  - "Helps with dry eyes" → low risk
  - "Improves cataracts" → high risk (surgery is the only validated treatment)

## Step 6: Assign Confidence

| Level | Criteria |
|-------|----------|
| ✅ verified | Multiple strong sources, consensus, creator credentials verified |
| ⚠️ plausible | Some evidence, logical mechanism, no red flags |
| 🔬 emerging | Real practice/therapy but evidence is preliminary or limited to tradition |
| ❌ debunked | Claim contradicts established evidence, mechanism doesn't hold |
| ❓ untested | No sources found, claim too vague to evaluate |

## When to note limitations

Always include in the "Nuances" section:
- Format limitations (Reel = no room for nuance)
- Missing information (guide not publicly accessible, method not described)
- Creator bias (selling a product/service, funnel marketing)
- What the claim leaves out (risks, contraindications, failure rates)

## Tools used

- `web_search` — multiple queries, different phrasings
- `web_extract` — for relevant landing pages, studies, reviews
- `curl + Googlebot UA` — Instagram metadata when web_extract is unavailable
- `yt-dlp + faster-whisper` — full Reel transcript for accurate claim extraction
