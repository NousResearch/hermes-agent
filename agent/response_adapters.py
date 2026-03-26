"""
Response adapters for models with non-standard output formats.

Converts model-specific response formats (like Qwen 3.5's <think> blocks)
into OpenAI-compatible message structures without modifying core agent logic.
"""

import html
import json
import re
from types import SimpleNamespace
from typing import Optional, List, Dict, Any


def _unescape_html_entities(text: str) -> str:
    """Unescape HTML entities like &quot; -> ", &apos; -> ' etc."""
    return html.unescape(text)


def _parse_python_code_blocks(content: str) -> tuple[List[Any], str]:
    """
    Parse Python code blocks that contain tool calls.
    
    Qwen sometimes outputs:
    ```python
    from hermes_tools import write_file
    write_file("path", "content")
    ```
    
    Returns:
        (tool_calls, remaining_content)
    """
    tool_calls = []
    
    # Split by ```python markers and process each block
    parts = content.split('```python')
    clean_parts = [parts[0]]  # Keep content before first code block
    
    for part in parts[1:]:
        # Find the end of this code block
        if '```' not in part:
            # Unclosed code block - keep as is
            clean_parts.append('```python' + part)
            continue
        
        code, _, after = part.partition('```')
        
        # Parse function calls from the code
        for line in code.strip().split('\n'):
            line = line.strip()
            # Skip import lines and empty lines
            if not line or line.startswith('from ') or line.startswith('import '):
                continue
            
            # Match function calls: name(arg1="val", ...)
            paren_idx = line.find('(')
            if paren_idx > 0 and line.endswith(')'):
                func_name = line[:paren_idx].strip()
                args_str = line[paren_idx+1:-1].strip()
                
                # Skip 'from' keyword
                if func_name == 'from':
                    continue
                
                # Parse arguments
                args = {}
                i = 0
                while i < len(args_str):
                    eq_idx = args_str.find('=', i)
                    if eq_idx == -1:
                        break
                    
                    key = args_str[i:eq_idx].strip()
                    if not key.isidentifier():
                        i = eq_idx + 1
                        continue
                    
                    # Find value start (quote after =)
                    val_start = eq_idx + 1
                    while val_start < len(args_str) and args_str[val_start] in ' \t':
                        val_start += 1
                    
                    if val_start >= len(args_str):
                        break
                    
                    quote = args_str[val_start]
                    if quote not in '"\'':
                        i = val_start + 1
                        continue
                    
                    val_end = args_str.find(quote, val_start + 1)
                    if val_end == -1:
                        break
                    
                    args[key] = args_str[val_start+1:val_end]
                    i = val_end + 1
                
                # Parse positional args if no kwargs found
                if not args and args_str:
                    strings = []
                    j = 0
                    while j < len(args_str):
                        if args_str[j] in '"\'':
                            quote = args_str[j]
                            end = args_str.find(quote, j+1)
                            if end != -1:
                                strings.append(args_str[j+1:end])
                                j = end + 1
                            else:
                                j += 1
                        else:
                            j += 1
                    
                    if len(strings) >= 2:
                        args['path'] = strings[0]
                        args['content'] = strings[1]
                    elif len(strings) == 1:
                        args['command'] = strings[0]
                
                if args:
                    tool_calls.append(SimpleNamespace(
                        id=f"call_{len(tool_calls)}",
                        type="function",
                        function=SimpleNamespace(
                            name=func_name,
                            arguments=json.dumps(args)
                        )
                    ))
        
        # Keep content after code block
        clean_parts.append(after.lstrip('\n'))
    
    clean_content = ''.join(clean_parts).strip()
    return tool_calls, clean_content


def _parse_hermes_json_tool_calls(content: str) -> tuple[List[Any], str]:
    """
    Parse Hermes-style JSON-in-XML format:
    <tool_call>{"name": "tool_name", "arguments": {"arg1": "val1"}}</tool_call>
    
    This is the OFFICIAL Hermes/Qwen standard format.
    See: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
    
    Args:
        content: Response content that may contain JSON tool calls
        
    Returns:
        (tool_calls, cleaned_content)
    """
    tool_calls = []
    
    # Pattern: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    json_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    
    matches = list(re.finditer(json_pattern, content, re.DOTALL))
    if not matches:
        return tool_calls, content
    
    for i, match in enumerate(matches):
        try:
            json_str = _unescape_html_entities(match.group(1))
            tool = json.loads(json_str)
            tool_call = _dict_to_tool_call(tool, i)
            if tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    # Remove parsed tool_call blocks
    cleaned_content = re.sub(json_pattern, '', content, flags=re.DOTALL).strip()
    return tool_calls, cleaned_content


def _parse_pure_xml_tool_calls(content: str) -> tuple[List[Any], str]:
    """
    Parse pure XML format (fallback for models that output this way):
    <tool_call>tool_name<arg1>val1</arg1><arg2>val2</arg2>...</tool_call>
    
    Some Qwen 3.5 MLX conversions output this format instead of Hermes JSON.
    
    Args:
        content: Response content that may contain XML tool calls
        
    Returns:
        (tool_calls, cleaned_content)
    """
    tool_calls = []
    
    # Pattern: <tool_call>tool_name<param>value</param>...</tool_call>
    # The tool_name comes first, then XML params
    xml_pattern = r'<tool_call>\s*(\w+)\s*(.*?)\s*</tool_call>'
    
    matches = list(re.finditer(xml_pattern, content, re.DOTALL))
    if not matches:
        return tool_calls, content
    
    for i, match in enumerate(matches):
        tool_name = match.group(1)
        params_block = match.group(2)
        
        # Skip if this looks like JSON (handled by _parse_hermes_json_tool_calls)
        if tool_name in ('{"name"', '{"function"'):
            continue
        
        # Parse XML-style params: <param_name>value</param_name>
        args = {}
        param_pattern = r'<(\w+)[^>]*>(.*?)</\1>|<(\w+)/>'
        
        for param_match in re.finditer(param_pattern, params_block, re.DOTALL):
            if param_match.group(3):  # Self-closing tag
                param_name = param_match.group(3)
                param_value = True
            else:
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                
                # Type inference
                if param_value.lower() == 'true':
                    param_value = True
                elif param_value.lower() == 'false':
                    param_value = False
                elif param_value.isdigit():
                    param_value = int(param_value)
                elif param_value.replace('.', '', 1).isdigit() and param_value.count('.') == 1:
                    param_value = float(param_value)
            
            args[param_name] = param_value
        
        if tool_name:
            tool_calls.append(SimpleNamespace(
                id=f"call_{i}",
                type="function",
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=json.dumps(args) if args else "{}"
                )
            ))
    
    cleaned_content = re.sub(xml_pattern, '', content, flags=re.DOTALL).strip()
    return tool_calls, cleaned_content


def adapt_qwen35_response(raw_content: str) -> Dict[str, Any]:
    """
    Parse Qwen 3.5 format: <think>reasoning</think>actual_content
    
    Qwen 3.5 uses special tokens for thinking mode:
    - Everything between <think> and </think> is internal reasoning
    - Everything after </think> is the actual response
    - Tool calls may be embedded in either section (XML format is native)
    
    Args:
        raw_content: The raw response string from the model
        
    Returns:
        Dict with keys:
        - content: The actual response content (after thinking)
        - reasoning: The reasoning/thinking content (if any)
        - tool_calls: List of SimpleNamespace objects (OpenAI format)
    """
    if not raw_content:
        return {"content": "", "reasoning": None, "tool_calls": []}
    
    # Pattern 1: Standard <think>...</think> format
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, raw_content, re.DOTALL)
    
    if think_match:
        reasoning = think_match.group(1).strip()
        # Content is everything after the closing </think> tag
        content = raw_content[think_match.end():].strip()
    else:
        # No think block found - treat entire content as response
        reasoning = None
        content = raw_content.strip()
    
    # Parse tool calls from content (and possibly reasoning)
    tool_calls = []
    
    # Format 0: Hermes JSON-in-XML (OFFICIAL standard, try first)
    # <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    hermes_tools, content = _parse_hermes_json_tool_calls(content)
    tool_calls.extend(hermes_tools)
    
    # Format 1: Pure XML format (fallback for some MLX conversions)
    # <tool_call>tool_name<arg1>val1</arg1><arg2>val2</arg2>...</tool_call>
    if not tool_calls:
        xml_tools, content = _parse_pure_xml_tool_calls(content)
        tool_calls.extend(xml_tools)
    
    # Format 2: Python code blocks
    if not tool_calls:
        py_tools, content = _parse_python_code_blocks(content)
        tool_calls.extend(py_tools)
    
    # Format 3: Qwen-Agent array format <tool_calls>[...]</tool_calls>
    if not tool_calls:
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tc_matches = re.findall(tool_call_pattern, content, re.DOTALL)
        if tc_matches:
            for tc_json in tc_matches:
                try:
                    tc_json_clean = _unescape_html_entities(tc_json)
                    tool = json.loads(tc_json_clean)
                    tool_calls.append(_dict_to_tool_call(tool, len(tool_calls)))
                except json.JSONDecodeError:
                    continue
            content = re.sub(tool_call_pattern, '', content, flags=re.DOTALL).strip()
    
    # Format 3: <tool_calls>[{"name": "...", "arguments": {...}}]</tool_calls>
    if not tool_calls:
        tool_calls_pattern = r'<tool_calls>\s*(\[.*?\])\s*</tool_calls>'
        tc_match = re.search(tool_calls_pattern, content, re.DOTALL)
        if tc_match:
            try:
                tools_json_str = _unescape_html_entities(tc_match.group(1))
                tools_json = json.loads(tools_json_str)
                tool_calls.extend(_normalize_tool_calls(tools_json))
                content = re.sub(tool_calls_pattern, '', content, flags=re.DOTALL).strip()
            except json.JSONDecodeError:
                pass
    
    # Format 4: JSON array directly in content
    if not tool_calls:
        json_pattern = r'\[\s*\{\s*"name"\s*:\s*"[^"]+".*?\}\s*\]'
        json_match = re.search(json_pattern, content, re.DOTALL)
        if json_match:
            try:
                tools_json = json.loads(json_match.group(0))
                tool_calls.extend(_normalize_tool_calls(tools_json))
                content = content.replace(json_match.group(0), '').strip()
            except json.JSONDecodeError:
                pass
    
    # Format 5: Parse intent from reasoning if no explicit tool_calls found
    if not tool_calls and reasoning:
        inferred_tools = _infer_tools_from_reasoning(reasoning)
        if inferred_tools:
            tool_calls.extend(inferred_tools)
    
    return {
        "content": content if content else None,
        "reasoning": reasoning,
        "tool_calls": tool_calls if tool_calls else None
    }


def _dict_to_tool_call(tool: Dict, index: int) -> Any:
    """Convert a tool dict to SimpleNamespace with OpenAI-compatible structure."""
    name = tool.get("name") or tool.get("function", {}).get("name")
    if not name:
        return None
    
    args = tool.get("arguments") or tool.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    
    return SimpleNamespace(
        id=f"call_{index}",
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args) if isinstance(args, dict) else str(args)
        )
    )


def _normalize_tool_calls(tools_json: List[Dict]) -> List[Any]:
    """
    Convert various tool call formats to OpenAI-compatible format.
    
    Supports:
    - {"name": "tool_name", "arguments": {...}}
    - {"name": "tool_name", "args": {...}}
    - {"function": {"name": "...", "arguments": "..."}}
    
    Returns:
        List of SimpleNamespace objects with attributes: id, type, function
    """
    tool_calls = []
    
    for i, tool in enumerate(tools_json):
        if not isinstance(tool, dict):
            continue
        tool_call = _dict_to_tool_call(tool, i)
        if tool_call:
            tool_calls.append(tool_call)
    
    return tool_calls


def _infer_tools_from_reasoning(reasoning: str) -> List[Any]:
    """
    Infer tool intent from reasoning text when explicit tool_calls aren't formatted.
    
    This helps weak models that think about using tools but don't output proper JSON.
    """
    tool_calls = []
    reasoning_lower = reasoning.lower()
    
    # Tool trigger patterns
    tool_patterns = {
        "terminal": [
            r'(?:use|run|execute|call)\s+(?:the\s+)?terminal\s+(?:tool\s+)?(?:to\s+)?(?:run|execute)?\s*["\']?([^"\'\n]{3,100})["\']?',
            r'(?:terminal|bash|shell)\s*(?:command)?:?\s*["\']?([^"\'\n]{3,100})["\']?',
        ],
        "read_file": [
            r'(?:read|check|open|view)\s+(?:the\s+)?(?:file\s+)?["\']?([\w\-\./]+\.[\w]{1,10})["\']?',
        ],
        "web_search": [
            r'(?:search|look up|google)\s+(?:for\s+)?["\']?([^"\'\n]{3,100})["\']?',
        ],
        "web_extract": [
            r'(?:extract|scrape|fetch)\s+(?:from\s+)?["\']?(https?://[^"\'\s]+)["\']?',
        ],
        "browser": [
            r'(?:browser|navigate|visit)\s+(?:to\s+)?["\']?(https?://[^"\'\s]+)["\']?',
        ],
    }
    
    for tool_name, patterns in tool_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, reasoning_lower)
            if match:
                arg_value = match.group(1).strip() if match.groups() else ""
                
                # Build appropriate arguments
                if tool_name == "terminal":
                    args = {"command": arg_value}
                elif tool_name == "read_file":
                    args = {"file_path": arg_value}
                elif tool_name == "web_search":
                    args = {"query": arg_value}
                elif tool_name in ["web_extract", "browser"]:
                    args = {"url": arg_value}
                else:
                    args = {}
                
                tool_calls.append(SimpleNamespace(
                    id=f"call_inferred_{len(tool_calls)}",
                    type="function",
                    function=SimpleNamespace(
                        name=tool_name,
                        arguments=json.dumps(args)
                    )
                ))
                break  # Only extract first match per tool
    
    return tool_calls[:3]  # Limit to 3 inferred tools max


def needs_adapter(model_name: str) -> bool:
    """
    Check if a model needs response adaptation.
    
    Args:
        model_name: The model identifier string
        
    Returns:
        True if the model uses a non-standard format
    """
    model_lower = model_name.lower()
    adapter_models = [
        "qwen3", "qwen-3",
        "kimi", "moonshot",
        # Add more as needed
    ]
    return any(pattern in model_lower for pattern in adapter_models)


def adapt_hermes_tool_call_response(raw_content: str) -> Dict[str, Any]:
    """
    Parse Hermes-style <tool_call> JSON format.
    
    Models like Kimi K2 and some Qwen variants output tool calls as:
    <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    
    This is handled by _parse_hermes_json_tool_calls.
    
    Args:
        raw_content: The raw response string from the model
        
    Returns:
        Dict with keys:
        - content: The actual response content (with tool_calls stripped)
        - reasoning: None (no thinking block in this format)
        - tool_calls: List of SimpleNamespace objects (OpenAI format)
    """
    if not raw_content:
        return {"content": "", "reasoning": None, "tool_calls": None}
    
    # Check if content has <tool_call> tags
    if "<tool_call>" not in raw_content:
        return {"content": raw_content, "reasoning": None, "tool_calls": None}
    
    # Use the existing hermes parser
    tool_calls, cleaned_content = _parse_hermes_json_tool_calls(raw_content)
    
    return {
        "content": cleaned_content if cleaned_content else None,
        "reasoning": None,
        "tool_calls": tool_calls if tool_calls else None
    }


def adapt_response(model_name: str, raw_content: str) -> Dict[str, Any]:
    """
    Main entry point - route to appropriate adapter based on model.
    
    Args:
        model_name: The model identifier
        raw_content: Raw response content from the model
        
    Returns:
        OpenAI-compatible response dict
    """
    model_lower = model_name.lower()
    
    if "qwen3" in model_lower or "qwen-3" in model_lower:
        return adapt_qwen35_response(raw_content)
    
    # Kimi models (and other models) that output <tool_call> JSON format
    if "kimi" in model_lower or "moonshot" in model_lower:
        return adapt_hermes_tool_call_response(raw_content)
    
    # Generic check: if content has <tool_call> tags, parse them
    if "<tool_call>" in raw_content:
        return adapt_hermes_tool_call_response(raw_content)
    
    # No adapter needed - return as-is
    return {
        "content": raw_content,
        "reasoning": None,
        "tool_calls": None
    }
