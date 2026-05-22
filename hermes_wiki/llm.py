from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

import openai

from hermes_wiki.config import WikiConfig

logger = logging.getLogger(__name__)


class WikiLLM:
    """GPT 5.5 integration for wiki intelligence operations."""

    def __init__(self, config: WikiConfig):
        self.config = config
        self._client = openai.OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )

    def _call(self, system: str, user: str, temperature: float | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature or self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _call_json(self, system: str, user: str) -> dict:
        raw = self._call(system, user)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response, returning raw text")
            return {"raw_response": raw}

    def analyze_source(self, source_text: str, existing_entities: list[str],
                       existing_concepts: list[str], schema_text: str) -> dict:
        """Analyze a source document to extract entities, concepts, key facts."""
        system = f"""You are a wiki analyst. Analyze source documents and extract structured information.

Today's date: {date.today().isoformat()}

Wiki schema:
{schema_text[:3000]}

Existing entity pages (use exact slugs when referencing): {', '.join(existing_entities[:100]) or 'none yet'}
Existing concept pages (use exact slugs when referencing): {', '.join(existing_concepts[:100]) or 'none yet'}

Return valid JSON:
{{
  "title": "Brief title for this source",
  "summary": "2-3 paragraph summary of key content",
  "entities": [
    {{"name": "Entity Name", "slug": "entity-name", "description": "One line", "is_new": true/false}}
  ],
  "concepts": [
    {{"name": "Concept Name", "slug": "concept-name", "description": "One line", "is_new": true/false}}
  ],
  "key_facts": ["Fact 1", "Fact 2"],
  "tags": ["tag1", "tag2"],
  "cross_references": ["existing-page-slug"],
  "contradictions": ["description of contradiction with existing knowledge"]
}}

IMPORTANT:
- Mark is_new=false if entity/concept matches an existing page slug
- Only create pages for entities/concepts CENTRAL to the source (not passing mentions)
- Passing mentions like "Markdown", "Knowledge Base", "hypertext" should NOT get pages unless
  the source discusses them substantively"""

        return self._call_json(system, f"Analyze this source document:\n\n{source_text[:12000]}")

    def generate_entity_page(self, entity: dict, source_summary: str,
                             source_path: str, existing_page_content: str | None,
                             related_context: str, known_pages: list[str]) -> str:
        """Generate or update an entity wiki page."""
        known_list = ", ".join(f"[[{p}]]" for p in known_pages[:80]) if known_pages else "none yet"

        system = f"""You are a wiki page writer. Write or update an entity page.

Today's date: {date.today().isoformat()}

CRITICAL RULES:
- Do NOT include YAML frontmatter (added separately)
- ONLY use [[wikilinks]] to pages that exist. Known pages: {known_list}
- Do NOT create wikilinks to pages that don't exist — use plain text instead
- Add provenance markers using the EXACT source path: ^[{source_path}]
- Start with a one-paragraph overview
- Be concise and factual — scannable in 30 seconds
- If updating existing content, preserve and extend it
- Use markdown: headers, bullet lists, bold for key terms"""

        user_parts = [f"Entity: {entity['name']}\nDescription: {entity.get('description', '')}"]
        user_parts.append(f"\nSource ({source_path}):\n{source_summary[:4000]}")
        if existing_page_content:
            user_parts.append(f"\nExisting page to update:\n{existing_page_content[:4000]}")
        if related_context:
            user_parts.append(f"\nRelated wiki pages:\n{related_context[:3000]}")

        body = self._call(system, "\n".join(user_parts))
        return self._sanitize_wikilinks(body, known_pages)

    def generate_concept_page(self, concept: dict, source_summary: str,
                              source_path: str, existing_page_content: str | None,
                              related_context: str, known_pages: list[str]) -> str:
        """Generate or update a concept wiki page."""
        known_list = ", ".join(f"[[{p}]]" for p in known_pages[:80]) if known_pages else "none yet"

        system = f"""You are a wiki page writer. Write or update a concept page.

Today's date: {date.today().isoformat()}

CRITICAL RULES:
- Do NOT include YAML frontmatter (added separately)
- ONLY use [[wikilinks]] to pages that exist. Known pages: {known_list}
- Do NOT create wikilinks to pages that don't exist — use plain text instead
- Add provenance markers using the EXACT source path: ^[{source_path}]
- Start with a clear definition/explanation
- Include: current state of knowledge, open questions, related concepts
- Be concise and factual
- If updating, integrate new information with existing content"""

        user_parts = [f"Concept: {concept['name']}\nDescription: {concept.get('description', '')}"]
        user_parts.append(f"\nSource ({source_path}):\n{source_summary[:4000]}")
        if existing_page_content:
            user_parts.append(f"\nExisting page to update:\n{existing_page_content[:4000]}")
        if related_context:
            user_parts.append(f"\nRelated wiki pages:\n{related_context[:3000]}")

        body = self._call(system, "\n".join(user_parts))
        return self._sanitize_wikilinks(body, known_pages)

    def _sanitize_wikilinks(self, body: str, known_pages: list[str]) -> str:
        """Replace wikilinks to unknown pages with plain text."""
        from hermes_wiki.frontmatter import slugify

        known_slugs = set(known_pages)

        def replace_link(match):
            raw = match.group(1)
            target = raw.split("|")[0].strip()
            display = raw.split("|")[1].strip() if "|" in raw else target
            slug = slugify(target)
            if slug in known_slugs:
                return match.group(0)
            return display

        return re.sub(r"\[\[([^\]]+)\]\]", replace_link, body)

    def generate_source_summary(self, source_text: str, source_name: str,
                                source_path: str) -> str:
        """Generate a summary page for a raw source."""
        system = f"""Write a concise summary of this source document.

Today's date: {date.today().isoformat()}

Structure:
- Brief overview (1-2 paragraphs)
- Key points (bullet list)
- Notable claims or data

Use provenance marker ^[{source_path}] for specific claims.
Do NOT include YAML frontmatter.
Do NOT use [[wikilinks]]."""

        return self._call(system, f"Source: {source_name}\n\n{source_text[:12000]}")

    def answer_query(self, question: str, wiki_context: str, search_results: str,
                     known_pages: list[str]) -> dict:
        """Answer a question using wiki context and search results."""
        known_list = ", ".join(known_pages[:60])

        system = f"""You are a wiki knowledge assistant. Answer questions using wiki content.

Today's date: {date.today().isoformat()}

Rules:
- Base your answer on the wiki content provided, not general knowledge
- Cite wiki pages using [[wikilinks]] — ONLY use these known pages: {known_list}
- If the wiki doesn't have enough information, say so
- Be direct and substantive

Return valid JSON:
{{
  "answer": "Your detailed answer with [[wikilinks]] citations",
  "sources_consulted": ["page-slug-1", "page-slug-2"],
  "confidence": "high|medium|low",
  "worth_filing": true/false,
  "file_title": "Suggested title if worth_filing",
  "file_slug": "suggested-slug"
}}

Set worth_filing=true ONLY for substantial synthesis or comparison — not simple lookups."""

        user = f"Question: {question}\n\nWiki content:\n{wiki_context[:8000]}\n\nSearch results:\n{search_results[:4000]}"
        return self._call_json(system, user)

    def detect_contradictions(self, page_a_content: str, page_a_name: str,
                              page_b_content: str, page_b_name: str) -> list[dict]:
        """Detect contradictions between two wiki pages."""
        system = """Analyze two wiki pages for contradictions — claims that conflict.

Return valid JSON:
{
  "contradictions": [
    {
      "claim_a": "What page A says",
      "claim_b": "What page B says (conflicting)",
      "severity": "major|minor",
      "resolution_suggestion": "How to resolve"
    }
  ]
}

Return empty list if no conflicts."""

        user = f"Page: {page_a_name}\n{page_a_content[:4000]}\n\n---\n\nPage: {page_b_name}\n{page_b_content[:4000]}"
        return self._call_json(system, user).get("contradictions", [])
