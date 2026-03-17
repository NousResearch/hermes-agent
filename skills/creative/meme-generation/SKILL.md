---
name: meme-generation
description: Generate meme ideas from a topic by selecting a suitable meme template and producing funny, relatable captions.
version: 1.0.0
author: adanaleycio
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [creative, memes, humor, social-media]
    related_skills: [ascii-art]
    requires_toolsets: [terminal]
---

# Meme Generation

Generate meme concepts from a topic by choosing a fitting meme template and writing short, funny captions.

## When to Use

Use this skill when the user:
- wants to make a meme about a topic
- has a subject, situation, or frustration and wants a funny meme version
- asks for a relatable, sarcastic, or programmer-style meme idea
- wants caption ideas matched to a known meme format

Do not use this skill when:
- the user wants a full graphic editor workflow
- the request is for hateful, abusive, or targeted harassment content
- the user wants a random joke without meme structure

## Quick Reference

| Input | Meaning |
|---|---|
| topic | The main subject of the meme |
| tone | Optional style: relatable, programmer, sarcastic |
| language | Optional output language |

| Template | Best for |
|---|---|
| This is Fine | chaos, denial, pretending things are okay |
| Distracted Boyfriend | distraction, shifting priorities |
| Two Buttons | dilemma between two bad choices |
| Expanding Brain | escalating irony or absurd superiority |
| Drake Hotline Bling | rejecting one thing and approving another |
| Gru's Plan | a plan that fails midway |
| Woman Yelling at Cat | blame, misunderstanding, argument |
| Change My Mind | strong ironic opinion |

## Procedure

1. Read the user's topic and determine the core situation.
2. Infer the emotional pattern behind the topic:
   - chaos
   - distraction
   - dilemma
   - escalation
   - contradiction
   - failed plan
3. Choose the meme template that best matches that pattern.
4. Briefly explain why the template fits the topic.
5. Generate 3 short caption options.
6. Keep captions aligned with the structure of the chosen meme.
7. If the user requests programmer humor, prefer themes like debugging, deployments, meetings, deadlines, code review, technical debt, and production incidents.
8. If the user requests Turkish, write naturally in Turkish instead of translating word-for-word.
9. Ensure the meme structure matches the template format strictly (e.g., two-line for "This is Fine", comparison labels for "Distracted Boyfriend").
10. If multiple templates are suitable, choose the one with the clearest and most relatable structure.

## Pitfalls

- Do not choose templates randomly; match the structure of the joke.
- Do not make captions too long.
- Do not ignore the requested tone.
- Do not generate hateful, abusive, or personally targeted content.
- Do not explain the meme too much; keep output concise and usable.
- Do not force a meme format if the topic does not fit one clearly.

## Verification

The output is correct if:
- the chosen template clearly matches the topic structure
- the explanation is brief and sensible
- all 3 captions are short, readable, and actually meme-like
- the tone matches the user's request
- the result is usable as a meme draft without major rewriting
- the meme can be directly turned into an image without additional rewriting
