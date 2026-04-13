import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

logger = logging.getLogger(__name__)

class LLMHistoryParser:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_export_file(self, json_file_path: Path) -> int:
        """Reads a JSON file, detects provider, and processes all threads. Returns number of threads processed."""
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON {json_file_path}: {e}")
            return 0

        # Best-effort detection
        # ChatGPT Takeout is usually a list of conversations
        if isinstance(data, list) and len(data) > 0:
            if "mapping" in data[0]:
                return self._process_chatgpt_export(data, json_file_path)
            elif "chat_messages" in data[0]:
                return self._process_claude_export(data, json_file_path)
        
        logger.warning(f"Unrecognized LLM export format in {json_file_path.name}")
        return 0

    def _safe_filename(self, title: str) -> str:
        safe = re.sub(r'[^a-zA-Z0-9_\- ]', '', title).strip()
        if not safe:
            safe = "Untitled_Thread"
        return safe[:100]

    def _process_chatgpt_export(self, conversations: List[Dict[str, Any]], source_file: Path) -> int:
        processed_count = 0
        for conv in conversations:
            title = conv.get("title") or "Untitled Conversation"
            conv_id = conv.get("id") or str(conv.get("create_time", "unknown"))
            create_time = conv.get("create_time", 0)
            
            # Extract messages by traversing the mapping
            messages = self._extract_chatgpt_messages(conv)
            if not messages:
                continue
                
            safe_title = self._safe_filename(title)
            filename = f"ChatGPT - {safe_title}.md"
            out_file = self.output_dir / filename
            
            # Deduplication Check
            existing_count = 0
            if out_file.exists():
                existing_fm, _ = self._parse_frontmatter(out_file.read_text(encoding="utf-8"))
                # If it's a different thread with the same name, we should technically append a UUID to filename
                # but we'll use uuid to check if it's the exact same thread
                if existing_fm.get("uuid") == conv_id:
                    existing_count = existing_fm.get("message_count", 0)
                else:
                    # Naming collision for different thread, add UUID to filename and reset existing_count
                    filename = f"ChatGPT - {safe_title}_{conv_id[-6:]}.md"
                    out_file = self.output_dir / filename
                    if out_file.exists():
                        existing_fm, _ = self._parse_frontmatter(out_file.read_text(encoding="utf-8"))
                        existing_count = existing_fm.get("message_count", 0)

            # If we have new messages
            new_messages = messages[existing_count:]
            if not new_messages:
                continue # Nothing new
                
            md_content = self._format_messages_to_markdown(new_messages)
            
            if existing_count == 0:
                # Brand new file
                date_str = datetime.fromtimestamp(create_time).strftime("%Y-%m-%d") if create_time else datetime.now().strftime("%Y-%m-%d")
                frontmatter = {
                    "source": "chatgpt",
                    "uuid": conv_id,
                    "date": date_str,
                    "message_count": len(messages),
                    "tags": ["external_brain", "chatgpt"]
                }
                fm_str = yaml.dump(frontmatter, sort_keys=False)
                final_content = f"---\n{fm_str}---\n\n# {title}\n\n{md_content}"
                out_file.write_text(final_content, encoding="utf-8")
            else:
                # Append to existing
                content = out_file.read_text(encoding="utf-8")
                # Update message count in frontmatter
                content = re.sub(
                    r"(message_count:\s*)\d+", 
                    r"\g<1>" + str(len(messages)), 
                    content, 
                    count=1
                )
                final_content = content + "\n\n" + md_content
                out_file.write_text(final_content, encoding="utf-8")
                
            processed_count += 1
            
        return processed_count

    def _extract_chatgpt_messages(self, conv: Dict[str, Any]) -> List[Tuple[str, str]]:
        mapping = conv.get("mapping", {})
        messages = []
        
        # Traverse from root (or simply sort by create_time if available on all nodes)
        nodes = []
        for node_id, node in mapping.items():
            msg = node.get("message")
            if msg and msg.get("author", {}).get("role") in ("user", "assistant"):
                nodes.append(msg)
                
        # Sort by creation time to reconstruct linear chat history
        nodes.sort(key=lambda x: x.get("create_time") or 0)
        
        for msg in nodes:
            role = msg.get("author", {}).get("role", "unknown").capitalize()
            parts = msg.get("content", {}).get("parts", [])
            # Some parts are dicts (e.g. tool calls) in newer exports, we just handle strings or convert to str
            text_parts = [p for p in parts if isinstance(p, str)]
            if text_parts:
                messages.append((role, "\\n".join(text_parts)))
                
        return messages

    def _format_messages_to_markdown(self, messages: List[Tuple[str, str]]) -> str:
        lines = []
        for role, text in messages:
            lines.append(f"### {role}")
            lines.append(text.strip())
            lines.append("---")
        return "\n\n".join(lines)

    def _process_claude_export(self, conversations: List[Dict[str, Any]], source_file: Path) -> int:
        processed_count = 0
        for conv in conversations:
            title = conv.get("name") or "Untitled Conversation"
            conv_id = conv.get("uuid") or str(conv.get("created_at", "unknown"))
            create_time_str = conv.get("created_at", "")
            
            raw_messages = conv.get("chat_messages", [])
            messages = []
            for msg in raw_messages:
                # 'human' or 'assistant'
                sender = msg.get("sender", "unknown")
                role = "User" if sender == "human" else "Assistant"
                text = msg.get("text", "")
                if text:
                    messages.append((role, text))
            
            if not messages:
                continue
                
            safe_title = self._safe_filename(title)
            filename = f"Claude - {safe_title}.md"
            out_file = self.output_dir / filename
            
            # Deduplication Check
            existing_count = 0
            if out_file.exists():
                existing_fm, _ = self._parse_frontmatter(out_file.read_text(encoding="utf-8"))
                if existing_fm.get("uuid") == conv_id:
                    existing_count = existing_fm.get("message_count", 0)
                else:
                    filename = f"Claude - {safe_title}_{conv_id[-6:]}.md"
                    out_file = self.output_dir / filename
                    if out_file.exists():
                        existing_fm, _ = self._parse_frontmatter(out_file.read_text(encoding="utf-8"))
                        existing_count = existing_fm.get("message_count", 0)

            new_messages = messages[existing_count:]
            if not new_messages:
                continue
                
            md_content = self._format_messages_to_markdown(new_messages)
            
            if existing_count == 0:
                # Try to parse ISO date string for frontmatter
                try:
                    date_str = create_time_str.split("T")[0] if create_time_str else datetime.now().strftime("%Y-%m-%d")
                except Exception:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    
                frontmatter = {
                    "source": "claude",
                    "uuid": conv_id,
                    "date": date_str,
                    "message_count": len(messages),
                    "tags": ["external_brain", "claude"]
                }
                fm_str = yaml.dump(frontmatter, sort_keys=False)
                final_content = f"---\n{fm_str}---\n\n# {title}\n\n{md_content}"
                out_file.write_text(final_content, encoding="utf-8")
            else:
                content = out_file.read_text(encoding="utf-8")
                content = re.sub(
                    r"(message_count:\s*)\d+", 
                    r"\g<1>" + str(len(messages)), 
                    content, 
                    count=1
                )
                final_content = content + "\n\n" + md_content
                out_file.write_text(final_content, encoding="utf-8")
                
            processed_count += 1
            
        return processed_count

    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if match:
            fm_text = match.group(1)
            body = content[match.end():]
            try:
                fm = yaml.safe_load(fm_text)
                return fm or {}, body
            except Exception:
                return {}, body
        return {}, content
