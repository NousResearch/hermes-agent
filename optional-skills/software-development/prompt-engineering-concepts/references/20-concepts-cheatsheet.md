# Source poster — "20 Prompt Engineering Concepts"

Akhil Meesala, *A Compact Cheatsheet with Ideas, Use Cases and Visualizations*
(Learn · Build · Grow). Transcribed verbatim below; the selection table in
`../SKILL.md` is the operational version of this list.

---

**1. Zero-Shot Prompting** — *Get results without examples.*
Visual: "Explain quantum computing simply."
★ Simple & direct prompts · ★ Best for common tasks · ★ Use clear instructions

**2. Few-Shot Prompting** — *Show examples for better results.*
Visual: Q: ...? A: ... / Q: ...? A: ...
★ Provide 1–5 examples · ★ Follow the pattern · ★ Helps with complex tasks

**3. Chain-of-Thought** — *Ask for step-by-step reasoning.*
Visual: "Let's think step by step…" → Plan → Reason → Answer
★ Break down the problem · ★ Improve reasoning · ★ Useful for math & logic

**4. Role Prompting** — *Assign a role for better tone.*
Visual: robot → "You are a senior data scientist…"
★ Set a role or persona · ★ Get expert responses · ★ Use domain expertise

**5. Instruction Prompting** — *Give clear, specific instructions.*
Visual: "Summarize this article in 3 bullet points."
★ Be explicit & detailed · ★ Specify format & tone · ★ Avoid ambiguity

**6. Output Formatting** — *Request structured output.*
Visual: `{"summary": "...", "key_points": [...]}`
★ Ask for JSON/structured · ★ Use schema if possible · ★ Improve consistency

**7. Temperature Control** — *Control randomness.*
Visual: slider 0.0 ——●—— 1.0
★ Lower for consistency · ★ Higher for creativity · ★ Tune per use case

**8. Prompt Chaining** — *Break tasks into steps.*
Visual: Plan → Research → Write
★ Split complex tasks · ★ Use intermediate steps · ★ Improve quality

**9. Self-Consistency** — *Get multiple answers.*
Visual: A1 / A2 / A3 → ✓ Final Answer
★ Run multiple prompts · ★ Select best response · ★ Reduce errors

**10. Retrieval-Augmented Generation (RAG)** — *Use external knowledge.*
Visual: document + magnifier + question → ✓
★ Retrieve relevant info · ★ Ground responses · ★ Reduce hallucinations

**11. Prompt Templates** — *Reuse proven templates.*
Visual: "You are a helpful assistant. Answer: {question}"
★ Create reusable templates · ★ Keep consistent style · ★ Save time

**12. Guardrails** — *Keep outputs safe & aligned.*
Visual: shield with ✓
★ Set safety rules · ★ Filter harmful content · ★ Ensure compliance
(badges: ✓ No hate · ✓ No harm · ✓ No PII)

**13. Dynamic Prompting** — *Adapt based on context.*
Visual: User Type? → Beginner / Expert / Pro
★ Personalize prompts · ★ Adjust complexity · ★ Improve engagement

**14. ReAct Prompting** — *Reason + Act for real tasks.*
Visual: Thought → Observation → Action
★ Think, take action · ★ Use tools effectively · ★ Solve real problems

**15. Tree-of-Thought** — *Explore multiple reasoning paths.*
Visual: Problem → Path 1 / Path 2 / Path 3
★ Explore options · ★ Choose best path · ★ Handle complex problems

**16. Prompt Compression** — *Reduce token usage.*
Visual: long doc → short doc
★ Remove redundancy · ★ Condense context · ★ Save costs

**17. Multi-Modal Prompts** — *Use text + images.*
Visual: [image] + "What is this?"
★ Combine text & images · ★ Get richer insights · ★ Build smarter apps

**18. JSON Mode** — *Get reliable JSON output.*
Visual: `{"answer": "..."}` ✓
★ Enable JSON mode · ★ Parse safely · ★ Use in workflows

**19. Prompt Evaluation** — *Measure and improve prompts.*
Visual: rising bar chart
★ Check quality & relevance · ★ Track success rate · ★ Iterate and improve

**20. Prompt Optimization** — *Iterate for best results.*
Visual: circular refresh arrows
★ Test variations · ★ Use metrics · ★ Optimize regularly
