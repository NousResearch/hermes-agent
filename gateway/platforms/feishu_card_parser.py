"""
飞书卡片解析器 - Feishu Card Parser
基于 OpenClaw CardConverter (TypeScript) 的 Python 实现

功能：
- 将飞书交互卡片 JSON 解析为结构化数据
- 支持 table、markdown、div、column_set 等所有主要元素
- 输出 markdown 格式文本

用法：
    from gateway.platforms.feishu_card_parser import FeishuCardParser
    
    parser = FeishuCardParser()
    result = parser.parse(card_json)
    print(result.markdown)       # markdown 格式文本
    print(result.title)          # 卡片标题
    print(result.tables)         # 表格列表
    print(result.actions)        # 按钮/交互元素
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TableData:
    """表格数据结构"""
    columns: list[str]
    rows: list[dict[str, str]]


@dataclass
class ActionData:
    """操作按钮数据结构"""
    tag: str  # button, select_static, overflow, etc.
    label: str
    value: Optional[str] = None


@dataclass
class ParseResult:
    """卡片解析结果"""
    title: str = ""
    markdown: str = ""
    text_content: str = ""
    tables: list[TableData] = field(default_factory=list)
    actions: list[ActionData] = field(default_factory=list)
    raw_json: Optional[dict] = None


class FeishuCardParser:
    """
    飞书交互卡片解析器
    
    将飞书 Schema 1.0 / 2.0 的卡片 JSON 解析为结构化数据。
    """
    
    # 支持的文本标签（在 rich block 内会提取文本）
    RICH_TEXT_TAGS = {
        "plain_text", "lark_md", "markdown", "markdown_v1",
        "note", "div", "column_set", "column", "action",
        "button", "select_static", "date_picker",
        "table", "hr", "br",
    }
    
    def __init__(self, mode: str = "concise"):
        """
        初始化解析器
        
        Args:
            mode: "concise" 或 "detailed" - 详细程度
        """
        self.mode = mode
    
    def parse(self, card_json: Any) -> ParseResult:
        """
        解析卡片 JSON
        
        Args:
            card_json: 卡片 JSON 对象（dict 或 JSON 字符串）
            
        Returns:
            ParseResult 对象
        """
        import json
        
        # 解析 JSON 字符串
        if isinstance(card_json, str):
            try:
                card_json = json.loads(card_json)
            except json.JSONDecodeError:
                return ParseResult(text_content="[无效的卡片 JSON]")
        
        if not isinstance(card_json, dict):
            return ParseResult(text_content="[卡片内容不是有效对象]")
        
        result = ParseResult(raw_json=card_json)
        
        # 提取 schema 版本
        schema = card_json.get("schema", 1)
        if isinstance(schema, str):
            schema = int(schema.replace("2.0", "2").replace("1.0", "1"))
        
        # 提取标题
        header = card_json.get("header") or card_json.get("card_header")
        if isinstance(header, dict):
            result.title = self._extract_header_title(header)
        
        # 提取 body
        body = card_json.get("body") or card_json.get("elements")
        if isinstance(body, dict):
            body = body.get("elements", [])
        
        if isinstance(body, list):
            # 处理嵌套数组情况: [[{...}, {...}]] -> [{...}, {...}]
            if len(body) == 1 and isinstance(body[0], list):
                body = body[0]
            self._convert_body(body, result)
        
        # 清理 markdown
        result.markdown = result.markdown.strip()
        result.text_content = result.markdown
        
        return result
    
    def _extract_header_title(self, header: dict) -> str:
        """从 header 提取标题"""
        prop = header.get("property") or header
        title = prop.get("title") or header.get("title")
        
        if isinstance(title, dict):
            return self._extract_text_content(title)
        elif isinstance(title, str):
            return title
        
        return ""
    
    def _convert_body(self, elements: list, result: ParseResult) -> None:
        """遍历 body elements 并转换"""
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            
            tag = (elem.get("tag") or elem.get("type") or "").lower()
            prop = elem.get("property") or elem
            
            if tag == "hr":
                result.markdown += "\n---\n"
            elif tag == "br":
                result.markdown += "\n"
            elif tag == "div":
                result.markdown += self._convert_div(prop) + "\n"
            elif tag == "markdown" or tag == "lark_md":
                result.markdown += self._convert_markdown(prop) + "\n"
            elif tag == "plain_text" or tag == "text":
                result.markdown += self._convert_plain_text(prop) + "\n"
            elif tag == "table":
                table_md = self._convert_table(prop)
                result.markdown += table_md + "\n"
                self._extract_table_data(prop, result)
            elif tag == "column_set":
                result.markdown += self._convert_column_set(prop) + "\n"
            elif tag == "note":
                result.markdown += self._convert_note(prop) + "\n"
            elif tag == "button":
                self._extract_action(elem, result, "button")
            elif tag == "select_static" or tag == "multi_select_static":
                self._extract_action(elem, result, "select_static")
            elif tag == "overflow":
                self._extract_action(elem, result, "overflow")
            elif tag == "date_picker" or tag == "picker_time" or tag == "picker_datetime":
                self._extract_action(elem, result, "date_picker")
            elif tag == "img" or tag == "image":
                result.markdown += self._convert_image(prop) + "\n"
            elif tag == "collapsible_panel":
                result.markdown += self._convert_collapsible_panel(prop) + "\n"
            elif tag == "heading":
                result.markdown += self._convert_heading(prop) + "\n"
            elif tag == "code_block":
                result.markdown += self._convert_code_block(prop) + "\n"
            elif tag == "blockquote":
                result.markdown += self._convert_blockquote(prop) + "\n"
            elif tag == "actions":
                self._convert_actions(prop, result)
    
    def _extract_text_content(self, elem: Any) -> str:
        """提取文本内容（支持 i18n）"""
        if not isinstance(elem, dict):
            return str(elem) if elem else ""
        
        # i18n 支持
        i18n = elem.get("i18nContent") or elem.get("i18n")
        if isinstance(i18n, dict):
            for lang in ["zh_cn", "en_us", "ja_jp"]:
                if i18n.get(lang):
                    return i18n[lang]
        
        # 直接 content
        content = elem.get("content") or elem.get("text")
        if isinstance(content, str):
            return content
        
        # elements 数组
        elements = elem.get("elements")
        if isinstance(elements, list) and elements:
            texts = []
            for e in elements:
                t = self._extract_text_content(e)
                if t:
                    texts.append(t)
            return "".join(texts)
        
        return ""
    
    def _apply_text_style(self, text: str, style: dict) -> str:
        """应用文本样式（粗体、斜体、删除线）"""
        if not text:
            return text
        
        attrs = style.get("attributes") or []
        if isinstance(attrs, list):
            if "strikethrough" in attrs:
                text = f"~~{text}~~"
            if "italic" in attrs:
                text = f"*{text}*"
            if "bold" in attrs:
                text = f"**{text}**"
        
        return text
    
    def _convert_div(self, prop: dict) -> str:
        """转换 div 元素"""
        results = []
        
        # text 字段
        text_elem = prop.get("text")
        if isinstance(text_elem, dict):
            tag = (text_elem.get("tag") or "").lower()
            if tag in ("markdown", "lark_md", "plain_text", "text"):
                text_content = self._extract_text_content(text_elem)
                style = text_elem.get("textStyle") or {}
                text_content = self._apply_text_style(text_content, style)
                results.append(text_content)
        
        # fields 数组（用于模拟表格的多列布局）
        fields = prop.get("fields")
        if isinstance(fields, list):
            field_texts = []
            for f in fields:
                if isinstance(f, dict):
                    te = f.get("text")
                    if isinstance(te, dict):
                        ft = self._extract_text_content(te)
                        style = te.get("textStyle") or {}
                        ft = self._apply_text_style(ft, style)
                        if ft:
                            field_texts.append(ft)
            if field_texts:
                results.append(" | ".join(field_texts))
        
        return "\n".join(results)
    
    def _convert_markdown(self, prop: dict) -> str:
        """转换 markdown 元素"""
        # elements 数组
        elements = prop.get("elements")
        if isinstance(elements, list) and elements:
            parts = []
            for elem in elements:
                if isinstance(elem, dict):
                    tag = (elem.get("tag") or "").lower()
                    if tag in ("plain_text", "text"):
                        content = self._extract_text_content(elem)
                        style = elem.get("textStyle") or {}
                        content = self._apply_text_style(content, style)
                        parts.append(content)
                    elif tag == "link":
                        content = self._extract_text_content(elem)
                        href = elem.get("href") or ""
                        if href:
                            parts.append(f"[{content}]({href})")
                        else:
                            parts.append(content)
                    elif tag == "emoji":
                        emoji_text = elem.get("emoji_type") or elem.get("unicode") or "📦"
                        parts.append(emoji_text)
                    elif tag == "at":
                        name = elem.get("name") or elem.get("value") or "@某人"
                        parts.append(f"@{name}")
                    elif tag == "code_span":
                        code = self._extract_text_content(elem)
                        parts.append(f"`{code}`")
                    else:
                        t = self._extract_text_content(elem)
                        if t:
                            parts.append(t)
            return "".join(parts)
        
        # 直接 content
        content = prop.get("content")
        if isinstance(content, str):
            return content
        
        return ""
    
    def _convert_plain_text(self, prop: dict) -> str:
        """转换纯文本元素"""
        content = prop.get("content") or prop.get("text") or ""
        style = prop.get("textStyle") or {}
        return self._apply_text_style(content, style)
    
    def _convert_table(self, prop: dict) -> str:
        """将表格转换为 markdown 格式"""
        columns = prop.get("columns")
        if not isinstance(columns, list) or len(columns) == 0:
            return ""
        
        rows = prop.get("rows") or []
        
        # 提取列名
        col_names = []
        col_keys = []
        for col in columns:
            if not isinstance(col, dict):
                continue
            display_name = col.get("display_name") or col.get("displayName") or col.get("name") or ""
            name = col.get("name") or ""
            col_names.append(str(display_name) if display_name else str(name))
            col_keys.append(str(name))
        
        if not col_names:
            return ""
        
        # 构建 markdown 表格
        lines = []
        lines.append("| " + " | ".join(col_names) + " |")
        lines.append("|" + "|".join("------" for _ in col_names) + "|")
        
        for row in rows:
            if not isinstance(row, dict):
                continue
            
            cells = []
            for key in col_keys:
                cell_data = row.get(key)
                cell_value = self._extract_table_cell_value(cell_data)
                cells.append(cell_value)
            
            lines.append("| " + " | ".join(cells) + " |")
        
        return "\n".join(lines)
    
    def _extract_table_cell_value(self, cell_data: Any) -> str:
        """提取表格单元格值"""
        if cell_data is None:
            return ""
        
        if isinstance(cell_data, str):
            return cell_data
        
        if isinstance(cell_data, (int, float)):
            return f"{cell_data:.2f}"
        
        if isinstance(cell_data, dict):
            # 新格式：直接字符串
            if "data" in cell_data:
                data = cell_data["data"]
                if isinstance(data, str):
                    return data
                elif isinstance(data, (int, float)):
                    return str(data)
                elif isinstance(data, list):
                    # 数组格式
                    texts = []
                    for item in data:
                        if isinstance(item, dict) and item.get("text"):
                            texts.append(f"「{item['text']}」")
                    return " ".join(texts)
                elif isinstance(data, dict):
                    return self._extract_text_content(data)
                return str(data)
            
            # 旧格式：直接从 dict 提取
            return self._extract_text_content(cell_data)
        
        return str(cell_data)
    
    def _extract_table_data(self, prop: dict, result: ParseResult) -> None:
        """提取表格数据到结构化对象"""
        columns = prop.get("columns")
        if not isinstance(columns, list) or len(columns) == 0:
            return
        
        rows = prop.get("rows") or []
        
        col_names = []
        col_keys = []
        for col in columns:
            if not isinstance(col, dict):
                continue
            display_name = col.get("display_name") or col.get("displayName") or col.get("name") or ""
            name = col.get("name") or ""
            col_names.append(str(display_name) if display_name else str(name))
            col_keys.append(str(name))
        
        if not col_names:
            return
        
        table = TableData(columns=col_names, rows=[])
        
        for row in rows:
            if not isinstance(row, dict):
                continue
            
            row_dict = {}
            for key, col_name in zip(col_keys, col_names):
                cell_data = row.get(key)
                row_dict[col_name] = self._extract_table_cell_value(cell_data)
            
            table.rows.append(row_dict)
        
        result.tables.append(table)
    
    def _convert_column_set(self, prop: dict, depth: int = 0) -> str:
        """转换 column_set（多列布局）"""
        columns = prop.get("columns") or prop.get("elements") or []
        results = []
        
        for col in columns:
            if not isinstance(col, dict):
                continue
            
            col_tag = (col.get("tag") or "").lower()
            col_prop = col.get("property") or col
            
            if col_tag == "column":
                elements = col_prop.get("elements") or []
                col_texts = []
                for elem in elements:
                    if isinstance(elem, dict):
                        tag = (elem.get("tag") or "").lower()
                        if tag == "div":
                            col_texts.append(self._convert_div(elem.get("property") or elem))
                        elif tag in ("markdown", "lark_md"):
                            col_texts.append(self._convert_markdown(elem.get("property") or elem))
                        elif tag in ("plain_text", "text"):
                            col_texts.append(self._convert_plain_text(elem.get("property") or elem))
                
                if col_texts:
                    results.append(" | ".join(col_texts))
            else:
                # 递归处理其他元素
                tag = (col.get("tag") or "").lower()
                if tag == "hr":
                    results.append("---")
                elif tag in ("markdown", "lark_md"):
                    results.append(self._convert_markdown(col.get("property") or col))
        
        return " ".join(results)
    
    def _convert_note(self, prop: dict) -> str:
        """转换 note 元素"""
        elements = prop.get("elements") or []
        texts = []
        
        for elem in elements:
            if isinstance(elem, dict):
                t = self._extract_text_content(elem)
                if t:
                    texts.append(t)
        
        return f"📝 {' '.join(texts)}" if texts else ""
    
    def _convert_image(self, prop: dict) -> str:
        """转换图片元素"""
        if self.mode == "detailed":
            img_key = prop.get("img_key") or prop.get("file_key") or ""
            return f"🖼️ 图片(key:{img_key})"
        return "🖼️ [图片]"
    
    def _convert_collapsible_panel(self, prop: dict) -> str:
        """转换可折叠面板"""
        header_text = ""
        header = prop.get("header") or {}
        if isinstance(header, dict):
            title_elem = header.get("title") or header.get("text")
            if isinstance(title_elem, dict):
                header_text = self._extract_text_content(title_elem)
            elif isinstance(title_elem, str):
                header_text = title_elem
        
        results = [f"### {header_text or '详情'}"]
        
        # 尝试获取展开内容（expanded 可能是 bool 或 dict）
        expanded = prop.get("expanded_content") or prop.get("expanded")
        if isinstance(expanded, dict):
            expand_elem = expanded.get("elements") or [expanded]
        elif isinstance(expanded, list):
            expand_elem = expanded
        else:
            expand_elem = []
        
        for elem in expand_elem:
            if isinstance(elem, dict):
                tag = (elem.get("tag") or "").lower()
                if tag == "markdown" or tag == "lark_md":
                    results.append(self._convert_markdown(elem.get("property") or elem))
                elif tag == "div":
                    results.append(self._convert_div(elem.get("property") or elem))
        
        return "\n".join(results)
    
    def _convert_heading(self, prop: dict) -> str:
        """转换标题元素"""
        content = self._extract_text_content(prop)
        level = prop.get("level", 2)
        prefix = "#" * min(level, 6)
        return f"{prefix} {content}"
    
    def _convert_code_block(self, prop: dict) -> str:
        """转换代码块"""
        content = self._extract_text_content(prop)
        language = prop.get("language") or ""
        return f"```{language}\n{content}\n```"
    
    def _convert_blockquote(self, prop: dict) -> str:
        """转换引用块"""
        content = self._extract_text_content(prop)
        lines = content.split("\n")
        return "\n".join(f"> {line}" for line in lines)
    
    def _extract_action(self, elem: dict, result: ParseResult, tag: str) -> None:
        """提取操作按钮"""
        prop = elem.get("property") or elem
        
        label = self._extract_text_content(prop.get("text") or prop.get("name") or prop)
        if not label:
            label = "按钮"
        
        value = prop.get("value") or prop.get("name") or ""
        
        result.actions.append(ActionData(tag=tag, label=label, value=str(value)))
    
    def _convert_actions(self, prop: dict, result: ParseResult) -> None:
        """转换 actions 容器内的所有操作"""
        actions = prop.get("actions") or prop.get("elements") or []
        
        for action in actions:
            if not isinstance(action, dict):
                continue
            
            tag = (action.get("tag") or "").lower()
            if tag in ("button", "select_static", "overflow", "date_picker"):
                self._extract_action(action, result, tag)


# 便捷函数
def parse_feishu_card(card_json: Any) -> ParseResult:
    """
    解析飞书卡片 JSON（便捷函数）
    
    Args:
        card_json: 卡片 JSON 对象或 JSON 字符串
        
    Returns:
        ParseResult 对象
    """
    parser = FeishuCardParser()
    return parser.parse(card_json)
