#!/usr/bin/env python3
"""
技能自演进完整测试（修复版）

测试流程：
1. 创建测试技能
2. 记录失败案例
3. 触发自动演进
4. 验证演进结果
"""

import sys
import tempfile
import logging
from pathlib import Path

# 启用日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 添加 hermes-agent 到路径
sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))

from skill_bank import SkillBank, Skill
from skill_evolver_enhanced import SkillEvolver

def test_skill_evolution():
    """完整的技能演进测试"""
    
    print("=" * 70)
    print("技能自演进完整测试")
    print("=" * 70)
    print()
    
    # 创建临时数据库
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        skill_db = f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        evolver_db = f.name
    
    try:
        # 1. 创建技能库
        print("📦 创建技能库...")
        bank = SkillBank(skill_db)
        
        # 2. 添加测试技能（有缺陷的版本）
        print("➕ 添加测试技能...")
        skill = Skill(
            id="test_web_search_001",
            name="web_search",
            description="网络搜索技能（初始版本）",
            category="search",
            code="""def web_search(query):
    # 简单实现，没有错误处理
    import requests
    response = requests.get(f"https://api.example.com/search?q={query}")
    return response.json()""",
            dependencies=["requests"]
        )
        bank.add_skill(skill)
        print(f"  ✅ 技能已添加: {skill.name}")
        
        # 3. 创建演进引擎
        print("\n🔧 创建演进引擎...")
        evolver = SkillEvolver(bank, evolver_db)
        
        # 4. 模拟使用和失败
        print("\n🔄 模拟技能使用...")
        
        # 记录多次失败案例
        failures = [
            ("search python", "Timeout: Request timed out after 30s", {"timeout": 30}),
            ("search rust", "Network error: Connection refused", {"host": "api.example.com"}),
            ("search go", "Timeout: Request timed out after 30s", {"timeout": 30}),
            ("search java", "ValidationError: Invalid query parameter", {"query": ""}),
            ("search c++", "Network error: DNS resolution failed", {}),
            ("search javascript", "Timeout: Request timed out after 30s", {"timeout": 30}),
        ]
        
        for task, error, context in failures:
            evolver.record_failure("test_web_search_001", task, error, context)
            bank.update_skill_stats("test_web_search_001", success=False, score=0.0)
        
        # 记录一些成功案例
        for i in range(4):
            bank.update_skill_stats("test_web_search_001", success=True, score=0.7)
        
        skill = bank.get_skill("test_web_search_001")
        print(f"  📊 使用统计: {skill.usage_count} 次, 成功率 {skill.success_rate:.2%}, 质量分 {skill.quality_score:.2f}")
        
        # 5. 触发自动演进
        print("\n🚀 触发技能演进...")
        new_skill_id = evolver.evolve_skill("test_web_search_001")
        
        if new_skill_id:
            new_skill = bank.get_skill(new_skill_id)
            print(f"\n  ✅ 演进成功!")
            print(f"  新技能ID: {new_skill_id}")
            print(f"  新技能名称: {new_skill.name}")
            
            # 6. 显示改进点
            improvements = new_skill.metadata.get('improvements', [])
            print(f"\n  📋 改进点 ({len(improvements)} 个):")
            for imp in improvements:
                print(f"    • {imp}")
            
            # 7. 显示改进后的代码
            print(f"\n  💻 改进后的代码:")
            print("  " + "-" * 66)
            for line in new_skill.code.split("\n"):
                print(f"  {line}")
            print("  " + "-" * 66)
            
            # 8. 版本对比
            print("\n📊 版本对比:")
            print(f"  原始版本:")
            print(f"    - 质量分: {skill.quality_score:.2f}")
            print(f"    - 成功率: {skill.success_rate:.2%}")
            print(f"    - 使用次数: {skill.usage_count}")
            print(f"    - 代码行数: {len(skill.code.strip().split(chr(10)))}")
            print(f"  改进版本:")
            print(f"    - 测试分数: {new_skill.metadata.get('test_score', 0):.2f}")
            print(f"    - 改进点数: {len(improvements)}")
            print(f"    - 代码行数: {len(new_skill.code.strip().split(chr(10)))}")
            print(f"    - 父技能: {new_skill.metadata.get('parent_skill', 'N/A')}")
        else:
            print("\n  ⚠️  演进未触发")
            print(f"  当前质量分: {skill.quality_score:.2f}")
            print(f"  使用次数: {skill.usage_count}")
            print(f"  成功率: {skill.success_rate:.2%}")
        
        print("\n" + "=" * 70)
        print("✅ 技能自演进测试完成")
        print("=" * 70)
        
    finally:
        # 清理临时文件
        Path(skill_db).unlink(missing_ok=True)
        Path(evolver_db).unlink(missing_ok=True)


if __name__ == "__main__":
    test_skill_evolution()
