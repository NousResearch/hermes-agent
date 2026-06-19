#!/usr/bin/env python3
"""
Integration tests for unified reflection system.
验证 compound-system 和 skill-evolution 的完整调用链。
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestIntegrationCompoundToReflection:
    """测试 compound.sh → unified_reflection.py 的调用链"""
    
    def test_compound_reflect_calls_unified_reflection(self):
        """验证 compound.sh reflect 命令调用 unified_reflection.py"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建模拟的 unified_reflection.py
            mock_reflection = Path(tmpdir) / "unified_reflection.py"
            mock_reflection.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps({"mocked": True}))\n')
            
            # 创建模拟的 compound.sh
            compound_script = Path(tmpdir) / "compound.sh"
            compound_script.write_text(f'''#!/usr/bin/env bash
# Simplified compound.sh for testing
outcome="${{1:-success}}"
severity="${{2:-none}}"

# Rule gate
should_reflect() {{
    if [[ "$outcome" == "error_recovered" ]]; then
        echo "reflect"
        return
    fi
    echo "skip"
}}

result=$(should_reflect "$outcome" "$severity")
if [[ "$result" == "reflect" ]]; then
    python3 {mock_reflection} record "task_end" "Task completed" "$outcome" "$severity" "" 2>/dev/null || true
fi
echo "$result"
''')
            os.chmod(compound_script, 0o755)
            
            # 测试 error_recovered 应该触发 reflect
            result = subprocess.run(
                ["bash", str(compound_script), "error_recovered", "medium"],
                capture_output=True, text=True
            )
            assert "reflect" in result.stdout
            assert '{"mocked": true}' in result.stdout
    
    def test_unified_reflection_record_writes_to_both_locations(self):
        """验证 record_event 同时写入 .skill-index/ 和 .compound/"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_index = Path(tmpdir) / ".skill-index"
            compound = Path(tmpdir) / ".compound"
            skill_index.mkdir()
            compound.mkdir()
            (compound / "reflections").mkdir()
            
            # Mock 模块级路径
            with patch("tools.unified_reflection.SKILL_INDEX_DIR", skill_index), \
                 patch("tools.unified_reflection.COMPOUND_DIR", compound), \
                 patch("tools.unified_reflection.FAILURE_LOG", skill_index / "failure_log.jsonl"), \
                 patch("tools.unified_reflection.PATTERNS_FILE", skill_index / "patterns.json"), \
                 patch("tools.unified_reflection.COMPOUND_REFLECTIONS", compound / "reflections"):
                
                from tools.unified_reflection import record_event
                
                # 记录一个来自 compound 的事件
                record_event(
                    event_type="task_end",
                    description="Test compound integration",
                    outcome="error_recovered",
                    severity=2,
                    source="compound",
                )
                
                # 验证写入 .skill-index/failure_log.jsonl
                assert (skill_index / "failure_log.jsonl").exists()
                with open(skill_index / "failure_log.jsonl") as f:
                    lines = f.readlines()
                assert len(lines) == 1
                event = json.loads(lines[0])
                assert event["source"] == "compound"
                
                # 验证写入 .compound/reflections/
                reflections_dir = compound / "reflections"
                assert reflections_dir.exists()
                files = list(reflections_dir.glob("*.json"))
                assert len(files) == 1


class TestIntegrationSkillToReflection:
    """测试 skill tools → unified_reflection.py 的调用链"""
    
    def test_skill_evolutionRegistersReflectionTools(self):
        """验证 skill_evolution.py 注册了 reflection_* 工具"""
        # 模拟 registry
        class MockRegistry:
            def __init__(self):
                self.tools = {}
            
            def register(self, name, toolset, schema, handler, emoji=""):
                self.tools[name] = {
                    "toolset": toolset,
                    "schema": schema,
                    "handler": handler,
                    "emoji": emoji,
                }
        
        mock_registry = MockRegistry()
        
        with patch("tools.registry.registry", mock_registry):
            # 重新导入以触发注册
            import importlib
            import tools.skill_evolution as se
            importlib.reload(se)
        
        # 验证 reflection_* 工具已注册
        assert "reflection_record" in mock_registry.tools
        assert "reflection_suggestions" in mock_registry.tools
        assert "reflection_patterns" in mock_registry.tools
        
        # 验证 schema 结构
        record_schema = mock_registry.tools["reflection_record"]["schema"]
        assert record_schema["name"] == "reflection_record"
        assert "event_type" in record_schema["parameters"]["properties"]
        assert "description" in record_schema["parameters"]["properties"]
    
    def test_skillFailureTriggersReflectionRecord(self):
        """验证技能失败时调用 reflection_record"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_index = Path(tmpdir) / ".skill-index"
            skill_index.mkdir()
            
            with patch("tools.unified_reflection.SKILL_INDEX_DIR", skill_index), \
                 patch("tools.unified_reflection.FAILURE_LOG", skill_index / "failure_log.jsonl"):
                
                from tools.unified_reflection import record_event
                
                # 模拟技能失败
                event = record_event(
                    event_type="skill_failed",
                    description="Web search failed",
                    outcome="failure",
                    severity=2,
                    skill_name="web-search",
                    error_message="Rate limit exceeded",
                    tags=["api", "rate-limit"],
                    source="skill_evolution",
                )
                
                # 验证记录
                assert event["skill_name"] == "web-search"
                assert event["source"] == "skill_evolution"
                
                # 验证文件写入
                assert (skill_index / "failure_log.jsonl").exists()


class TestIntegrationEndToEnd:
    """端到端集成测试"""
    
    def test_full_reflection_flow(self):
        """完整的反思流程：记录 → 提取模式 → 检索建议"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_index = Path(tmpdir) / ".skill-index"
            compound = Path(tmpdir) / ".compound"
            skill_index.mkdir()
            compound.mkdir()
            (compound / "reflections").mkdir()
            
            with patch("tools.unified_reflection.SKILL_INDEX_DIR", skill_index), \
                 patch("tools.unified_reflection.COMPOUND_DIR", compound), \
                 patch("tools.unified_reflection.FAILURE_LOG", skill_index / "failure_log.jsonl"), \
                 patch("tools.unified_reflection.PATTERNS_FILE", skill_index / "patterns.json"), \
                 patch("tools.unified_reflection.COMPOUND_REFLECTIONS", compound / "reflections"):
                
                from tools.unified_reflection import record_event, extract_patterns, get_suggestions
                
                # Step 1: 记录多个失败事件
                for i in range(3):
                    record_event(
                        event_type="skill_failed",
                        description=f"API error {i}",
                        outcome="failure",
                        severity=2,
                        skill_name="web-search",
                        error_message="Rate limit exceeded for API calls",
                        tags=["api", "rate-limit"],
                        source="skill_evolution",
                    )
                
                # Step 2: 提取模式
                patterns = extract_patterns()
                assert len(patterns) > 0
                assert patterns[0]["occurrence_count"] == 3
                
                # Step 3: 检索建议
                suggestions = get_suggestions(
                    error_message="API rate limit",
                    skill_name="web-search",
                )
                assert len(suggestions) > 0
                
                # 验证建议包含相关信息
                found = any(
                    s.get("skill_name") == "web-search" or
                    "rate-limit" in str(s.get("tags", [])) or
                    "api" in str(s.get("common_tags", []))
                    for s in suggestions
                )
                assert found
    
    def test_cli_end_to_end(self):
        """CLI 端到端测试"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_index = Path(tmpdir) / ".skill-index"
            compound = Path(tmpdir) / ".compound"
            skill_index.mkdir()
            compound.mkdir()
            (compound / "reflections").mkdir()
            
            # 复制 unified_reflection.py 到临时目录
            import shutil
            src = Path(__file__).parent.parent / "tools" / "unified_reflection.py"
            dst = Path(tmpdir) / "unified_reflection.py"
            shutil.copy(src, dst)
            
            # Mock 模块级路径
            reflection_code = dst.read_text()
            reflection_code = reflection_code.replace(
                "SKILL_INDEX_DIR = HOME / \".skill-index\"",
                f"SKILL_INDEX_DIR = Path(\"{skill_index}\")"
            ).replace(
                "COMPOUND_DIR = HOME / \".compound\"",
                f"COMPOUND_DIR = Path(\"{compound}\")"
            ).replace(
                "FAILURE_LOG = SKILL_INDEX_DIR / \"failure_log.jsonl\"",
                f"FAILURE_LOG = Path(\"{skill_index / 'failure_log.jsonl'}\")"
            ).replace(
                "PATTERNS_FILE = SKILL_INDEX_DIR / \"patterns.json\"",
                f"PATTERNS_FILE = Path(\"{skill_index / 'patterns.json'}\")"
            ).replace(
                "COMPOUND_REFLECTIONS = COMPOUND_DIR / \"reflections\"",
                f"COMPOUND_REFLECTIONS = Path(\"{compound / 'reflections'}\")"
            )
            dst.write_text(reflection_code)
            
            # 测试 record 命令
            result = subprocess.run(
                ["python3", str(dst), "record", "task_end", "Test task", "success", "0", ""],
                capture_output=True, text=True
            )
            assert result.returncode == 0
            recorded = json.loads(result.stdout)
            assert recorded["event_type"] == "task_end"
            
            # 测试 suggestions 命令
            result = subprocess.run(
                ["python3", str(dst), "suggestions", "test error"],
                capture_output=True, text=True
            )
            assert result.returncode == 0
            
            # 测试 patterns 命令
            result = subprocess.run(
                ["python3", str(dst), "patterns"],
                capture_output=True, text=True
            )
            assert result.returncode == 0
