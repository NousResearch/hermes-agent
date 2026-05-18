#!/usr/bin/env python3
"""
Prompt Crafter CLI — analyze, optimize, and generate AI prompts.

Usage:
    python3 prompt_crafter.py analyze "You are a code reviewer..."
    python3 prompt_crafter.py templates
    python3 prompt_crafter.py templates --name code-review
    python3 prompt_crafter.py variations "Explain quantum computing"

No API keys required. Uses heuristic-based analysis.
"""

import json
import sys

QUALITY_CHECKS = [
    {"id": "has_role", "name": "Role/Persona", "desc": "Clear role assignment (You are a...)",
     "check": lambda p: any(p.lower().startswith(phrase) or f" {phrase}" in p.lower()
                           for phrase in ["you are", "act as", "you're a", "as a professional"])},
    {"id": "has_context", "name": "Context", "desc": "Background context before instructions",
     "check": lambda p: len(p.split()) > 30},
    {"id": "has_constraints", "name": "Constraints", "desc": "Boundaries (limits, exclusions, rules)",
     "check": lambda p: any(w in p.lower() for w in ["don't", "do not", "avoid", "limit", "maximum",
                                                       "minimum", "only", "strictly", "never", "must not"])},
    {"id": "has_examples", "name": "Examples (Few-shot)", "desc": "Sample inputs/outputs included",
     "check": lambda p: any(w in p.lower() for w in ["example", "for instance", "e.g.", "like this",
                                                       "sample", "for example"])},
    {"id": "has_output_format", "name": "Output Format", "desc": "Exact output structure specified",
     "check": lambda p: any(w in p.lower() for w in ["output", "return", "format", "json", "markdown",
                                                       "bullet", "list:", "as json", "as markdown"])},
    {"id": "has_goal", "name": "Clear Goal", "desc": "Specific, measurable objective",
     "check": lambda p: len(p.split()) > 10},
    {"id": "has_tone", "name": "Tone/Style", "desc": "Communication style defined",
     "check": lambda p: any(w in p.lower() for w in ["tone", "style", "professional", "casual",
                                                       "friendly", "formal", "humorous", "concise"])},
    {"id": "has_cot", "name": "Chain of Thought", "desc": "Step-by-step reasoning encouraged",
     "check": lambda p: any(w in p.lower() for w in ["step by step", "think step", "reason step",
                                                       "explain your reasoning", "think through"])},
]

TEMPLATES = {
    "code-review": {
        "name": "Code Review",
        "prompt": "You are a senior software engineer reviewing code. Review the following {language} code:\n\n{code}\n\nFocus on:\n1. Security vulnerabilities\n2. Performance issues\n3. Code quality\n4. Edge cases\n5. Best practices\n\nFormat: **Critical** | **Warning** | **Suggestion**"
    },
    "explain-like-im-5": {
        "name": "Explain Like I'm 5",
        "prompt": "Explain {topic} in simple terms. Use analogies from everyday life. No jargon. Keep it under 3 paragraphs. Start with the simplest explanation."
    },
    "brainstorm": {
        "name": "Brainstorm Ideas",
        "prompt": "Generate ideas about {topic}. Requirements:\n- 10-15 diverse ideas\n- 1-sentence description each\n- Include unconventional approaches\n- Pick top 3 and explain why"
    },
}


def analyze_prompt(prompt: str) -> dict:
    wc = len(prompt.split())
    results = []
    passed = 0
    for c in QUALITY_CHECKS:
        ok = c["check"](prompt)
        if ok:
            passed += 1
        results.append({"check": c["name"], "passed": ok, "description": c["desc"]})

    score = int((passed / len(QUALITY_CHECKS)) * 100)
    suggestions = [f"Ekle: {r['description']}" for r in results if not r["passed"]]

    if score < 40:
        verdict = "Zayif prompt, yeniden yazilmali"
    elif score < 60:
        verdict = "Ortalama prompt, gelistirilmeli"
    elif score < 80:
        verdict = "Iyi prompt, kucuk iyilestirmeler mumkun"
    else:
        verdict = "Guclu prompt"

    return {
        "word_count": wc,
        "quality_score": score,
        "checks_passed": passed,
        "checks_total": len(QUALITY_CHECKS),
        "details": results,
        "suggestions": suggestions,
        "verdict": verdict,
    }


def get_template(name: str = None) -> dict:
    if name:
        t = TEMPLATES.get(name)
        if not t:
            return {"error": f"Template '{name}' bulunamadi. Mevcut: {list(TEMPLATES.keys())}"}
        return {"template": t["name"], "content": t["prompt"]}
    return {"available_templates": [{"id": k, "name": v["name"]} for k, v in TEMPLATES.items()]}


def generate_variations(prompt: str) -> list:
    analysis = analyze_prompt(prompt)
    vars_list = []

    if not any(r["passed"] for r in analysis["details"] if r["check"] == "Role/Persona"):
        vars_list.append(("Role Ekle", f"You are an expert assistant specialized in this field.\n\n{prompt}"))

    if not any(r["passed"] for r in analysis["details"] if r["check"] == "Output Format"):
        vars_list.append(("Format Belirt", f"{prompt}\n\nProvide your response in a clear, structured format."))

    if not any(r["passed"] for r in analysis["details"] if r["check"] == "Constraints"):
        vars_list.append(("Kisitlama Ekle", f"{prompt}\n\nBe concise and specific. Avoid unnecessary details."))

    if not vars_list:
        vars_list.append(("Orijinal (guclu)", prompt))

    return [{"style": v[0], "prompt": v[1]} for v in vars_list]


def cmd_analyze(args):
    text = " ".join(args.text) if args.text else ""
    if not text:
        text = sys.stdin.read().strip()
    print(json.dumps(analyze_prompt(text), indent=2, ensure_ascii=False))


def cmd_templates(args):
    print(json.dumps(get_template(args.name), indent=2, ensure_ascii=False))


def cmd_variations(args):
    text = " ".join(args.text) if args.text else ""
    if not text:
        text = sys.stdin.read().strip()
    print(json.dumps({"original": text, "variations": generate_variations(text)}, indent=2, ensure_ascii=False))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "analyze":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("text", nargs="*")
        cmd_analyze(p.parse_args(sys.argv[2:]))
    elif cmd == "templates":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--name")
        cmd_templates(p.parse_args(sys.argv[2:]))
    elif cmd == "variations":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("text", nargs="*")
        cmd_variations(p.parse_args(sys.argv[2:]))
    elif cmd in ("--help", "-h"):
        print(__doc__)
    else:
        print(f"Bilinmeyen komut: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()