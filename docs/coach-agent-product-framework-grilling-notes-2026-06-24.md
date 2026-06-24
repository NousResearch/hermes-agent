# Coach-Agent Product Framework — Grilling Notes

**Date:** 2026-06-24
**Status:** Alignment draft captured for repo ingestion
**Purpose:** Document the question-by-question grilling session that established the aligned product framework for coach-agent hatching.

---

## Summary

This document captures the locked decisions from a structured grilling session about the **Coach-Agent Product Framework**.

Primary conclusions:
- We are standardizing the **coach-agent product framework first**, with **Darin** as the first serious instantiation and **Erika** as a later instantiation.
- The platform is designed to **amplify the coach, not replace the coach**.
- The initial mandatory path is:
  1. Identity
  2. Brand Design Doc
  3. Teaching Intelligence
  4. Review / Launch Readiness
  5. Live Lane
- **Practice Outputs** are moved out of the mandatory path and become an **optional validation lane**.
- The **first live lane** is narrowly scoped to **audio-grounded lesson recap / follow-up packet generation**.
- New coach agents should start with **The System branding by default** unless and until the coach replaces it.

---

## Locked decisions

### Q1
**Question:** What is the primary thing we are standardizing here: a coach-agent product framework, or a Darin-first hatch plan that later gets generalized?

**Answer:** Coach-agent product framework first.

**Locked decision:** Standardize the **coach-agent product framework first**, with Darin as the first serious instantiation.

### Q2
**Question:** Should clean-start / contamination control be an explicit framework Phase 0?

**Answer:** No. That is a Darin-specific one-off remediation issue, not part of the normal framework.

**Locked decision:** **No general Phase 0.**

### Q3
**Question:** Should Identity and Brand Design Doc be hard-separated phases, or merged because they overlap?

**Answer:** Hard-separate them. Identity should come from an easy post-hatch introductory exchange; Brand should be a distinct explicit branding exercise producing a reusable document.

**Locked decision:** **Identity and Brand are hard-separated.**

### Q4
**Question:** Should identity be primarily discovered from the coach in conversation, or primarily designed by operator/product and then lightly personalized?

**Answer:** Primarily discovered from the coach in conversation. The only things we want to pre-load are a set of skills.

**Locked decision:** **Pre-load capabilities, discover identity.**

### Q5
**Question:** Should those pre-loaded skills be visible to the coach as explicit product capabilities during hatch, or mostly invisible infrastructure that the coach just experiences naturally through conversation?

**Answer:** Coaches should know their agent has capabilities right out of the box, but it cannot feel technical.

**Locked decision:** **Coach-facing capability visibility, without technical language.**

### Q6
**Question:** Should the introductory post-hatch identity conversation happen before the coach sees concrete example capabilities, or should the agent lead with a few immediately understandable things it can already help with?

**Answer:** Lead with a few immediately understandable capabilities first.

**Locked decision:** **Capability reassurance first, then identity discovery.**

### Q7
**Question:** Should the introductory identity conversation be a single lightweight onboarding chat, or an ongoing identity-shaping process over the first several interactions?

**Answer:** It should be a starting point for the identity.

**Locked decision:** **Identity begins with the onboarding chat, then can refine lightly through early interactions.**

### Q8
**Question:** Should the Brand Design Doc be required before meaningful practice outputs, or can the agent begin producing practice outputs with a provisional default style before branding is fully defined?

**Answer:** The coach can bypass custom brand design and default to **The System** branded docs.

**Locked decision:** **Brand customization is optional; default The System branding is the fallback.**

### Q9
**Question:** Should every new coach agent automatically start with The System branding by default unless and until the coach explicitly replaces it?

**Answer:** Yes.

**Locked decision:** **Default brand = The System, until explicitly replaced.**

### Q10
**Question:** What is the real purpose of the Teaching Intelligence phase: is it mainly to define how the agent interprets lesson material, or mainly to define what makes its outputs genuinely useful to the coach?

**Answer:** Mainly what makes outputs genuinely useful to the coach. If we are not useful for the coach, we will not have customers.

**Locked decision:** **Teaching Intelligence is primarily a coach-usefulness model, with lesson interpretation in service of that.**

### Q11
**Question:** Should Teaching Intelligence be standardized as one shared core model across all coach agents, or should each coach’s agent substantially adapt its usefulness model to that coach’s style and preferences?

**Answer:** Shared core model first, with coach-specific adaptation layered on top. Also, we should listen carefully to early coaches about whether that is enough for meaningful output in their voice.

**Locked decision:** **Shared teaching-intelligence core, coach-specific tuning, informed by early coach feedback.**

### Q12
**Question:** Should “the agent is not the coach” be treated as a core product doctrine by default, or only as a Bryan/Darin-specific preference?

**Answer:** Core product doctrine by default. We build a platform to amplify the coach, not replace the coach.

**Locked decision:** **Core doctrine: the platform amplifies the coach; it does not replace the coach.**

### Q13
**Question:** If the agent is not the coach, should the Practice Outputs phase optimize primarily for preserving and packaging the coach’s actual lesson voice, rather than generating new “helpful” teaching content?

**Answer:** Yes.

**Locked decision:** **If used, practice outputs should optimize for faithful preservation and digestible packaging of the coach’s instruction.**

### Q14
**Question:** Should the first Live Lane be narrowly limited to lesson recap / follow-up packet generation from lesson audio, rather than broader coach-assistant behaviors?

**Answer:** Yes. Future options may include calendar sync, student roster import, etc.

**Locked decision:** **First live lane = audio-grounded lesson recap / follow-up packet generation.**

### Q15
**Question:** Should future capabilities like calendar sync and roster import be part of the same hatch framework, or treated as separate post-hatch capability expansions once the core recap lane is trusted?

**Answer:** Treat them as separate to start.

**Locked decision:** **Future capabilities are separate post-hatch expansions.**

### Q16
**Question:** Should practice outputs be required before live use, including when a coach has no recorded lessons yet?

**Answer:** No. It’s great if the coach has material and we should ask and test on it, but coaches should be able to jump in without any such materials.

**Locked decision:** **Practice outputs are optional, not required.**

### Q17
**Question:** Given Q16, should Practice Outputs stop being a formal mandatory hatch phase and instead become an optional validation lane?

**Answer:** Yes. Move it out of the mandatory core path.

**Locked decision:** **Practice Outputs becomes an optional validation lane, not a required hatch phase.**

### Q18
**Question:** Given that coaches should be able to jump in quickly, should Review / Launch Readiness be a lightweight internal/operator gate rather than a heavy coach-visible stage?

**Answer:** Yes. Keep it lightweight.

**Locked decision:** **Review / Launch Readiness should be lightweight and mostly internal.**

### Q19
**Question:** Should the Brand Design Doc usually happen immediately during onboarding, or should it be allowed to happen later after the coach has already started using the agent under default The System branding?

**Answer:** Allow it to happen later, but the question must be asked in the initial flow. The agent should explain that the coach can skip it and default to The System branding. The coach should also be able to redo the brand design process any time. Erika also needs a co-branded-docs use case, where her brand is primary and a partner club brand is complementary.

**Locked decision:**
- **Brand question is mandatory in the initial flow**
- **Custom brand setup is optional and can be deferred**
- **Default fallback is The System branding**
- **Brand design can be redone later**
- **Framework must support co-branded brand design docs**

### Q20
**Question:** Should co-branding be treated as a standard feature of the Brand Design Doc framework for all coach agents, or as an advanced option we only surface when a coach actually needs it?

**Answer:** Standard framework support, but advanced/conditional surfacing.

**Locked decision:** **Co-branding is supported in the framework by default, but only surfaced when relevant.**

### Q21
**Question:** In the initial onboarding flow, should the agent explicitly explain the doctrine that “I’m here to amplify your coaching, not replace you,” or should that stay mostly implicit in the behavior and outputs?

**Answer:** Yes — lightly explicit.

**Locked decision:** **The onboarding flow should state this principle once in simple, natural language, then reinforce it through behavior.**

### Q22
**Question:** In the very first onboarding exchange, should the agent ask mostly open-ended “tell me about you” questions, or use more guided prompts that make it easy for the coach to respond without feeling put on the spot?

**Answer:** Use gentle guided prompts.

**Locked decision:** **The onboarding exchange should use gentle guided prompts, with room for open-ended elaboration.**

### Q23
**Question:** Should the initial onboarding flow aim to finish in one short conversation, or is it acceptable for identity + brand choice + first useful interaction to unfold across multiple lightweight turns if that feels more natural?

**Answer:** Multiple lightweight turns is better than forcing one perfect intake.

**Locked decision:** **The onboarding flow may unfold across multiple lightweight turns rather than a single rigid intake.**

---

## Resulting framework shape

### Mandatory core path
1. **Identity**
2. **Brand Design Doc**
3. **Teaching Intelligence**
4. **Review / Launch Readiness**
5. **Live Lane**

### Optional validation lane
- **Practice Outputs**

### First live lane
- **Audio-grounded lesson recap / follow-up packet generation**

### Branding modes
1. **Default The System brand**
2. **Coach-primary custom brand**
3. **Coach-primary co-branded mode** with partner/club complement when relevant

---

## Truth boundaries

What this document proves:
- The question-by-question alignment decisions from the grilling session were captured in repo-local markdown.
- The framework direction is documented in a James-ingestable format.

What this document does not yet prove:
- Finalized canonical framework wording for the main repo
- Implementation of the onboarding flow
- Live runtime enforcement of the framework
- Final The System Brand Design Doc artifact

---

## Next suggested repo artifacts

1. **Coach-Agent Product Framework v2**
2. **The System Brand Design Document**
3. **Darin instantiation notes**
4. **Erika instantiation notes**
