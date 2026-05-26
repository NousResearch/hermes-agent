#!/usr/bin/env python3
"""
核心改造管理工具

提供便捷的命令行接口来管理和使用核心改造的所有功能
"""

import sys
import os
import json
import argparse
from datetime import datetime

# 添加 Hermes agent 路径
sys.path.insert(0, os.path.expanduser('~/.hermes/hermes-agent'))

def cmd_status():
    """显示核心改造状态"""
    print("=" * 70)
    print("核心改造状态")
    print("=" * 70)
    
    # 检查模块安装
    print("\n📦 已安装模块:")
    
    modules = [
        ('hermes_context_engine', 'Context Engine'),
        ('hermes_multi_model_router', '多模型路由器'),
    ]
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} (未安装)")
    
    # 检查主链路插桩
    try:
        from agent.trace_hooks import trace_tool_call
        print(f"  ✅ 主链路插桩")
    except ImportError:
        print(f"  ❌ 主链路插桩 (未安装)")
    
    # 检查轨迹统计
    print("\n📊 轨迹统计:")
    try:
        from agent.trace_hooks import get_trace_stats
        stats = get_trace_stats()
        if stats:
            print(f"  总数: {stats['total']}")
            print(f"  成功: {stats['success']}")
            print(f"  失败: {stats['failed']}")
            print(f"  平均耗时: {stats['avg_duration_ms']:.2f}ms")
        else:
            print("  暂无轨迹数据")
    except Exception as e:
        print(f"  ❌ 无法获取统计: {e}")
    
    print()

def cmd_stats():
    """显示详细统计"""
    print("=" * 70)
    print("轨迹详细统计")
    print("=" * 70)
    
    try:
        from agent.trace_hooks import get_trace_stats
        stats = get_trace_stats()
        
        if not stats:
            print("\n暂无轨迹数据")
            print("提示: 运行一些 Hermes 任务后会自动积累轨迹")
            return
        
        print(f"\n总轨迹数: {stats['total']}")
        print(f"成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"失败: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
        print(f"平均耗时: {stats['avg_duration_ms']:.2f}ms")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def cmd_export(output_path):
    """导出轨迹"""
    print("=" * 70)
    print("导出轨迹")
    print("=" * 70)
    
    try:
        from agent.trace_hooks import export_traces
        
        # 如果没有指定路径，使用默认路径
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.expanduser(f'~/.hermes/traces/traces_{timestamp}.json')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        export_traces(output_path)
        print(f"\n✅ 轨迹已导出到: {output_path}")
        
        # 显示文件大小
        size = os.path.getsize(output_path)
        print(f"文件大小: {size:,} 字节")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def cmd_test_context_engine():
    """测试 Context Engine"""
    print("=" * 70)
    print("测试 Context Engine")
    print("=" * 70)
    
    try:
        import hermes_context_engine as hce
        
        # 测试 Tokenizer
        print("\n🧪 测试 Tokenizer...")
        tokenizer = hce.PyTokenizer("gpt-4")
        
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "人工智能正在改变世界。",
        ]
        
        for text in test_texts:
            count = tokenizer.count_text(text)
            print(f"  '{text}' = {count} tokens")
        
        # 测试 CoordinateTracker
        print("\n🧪 测试 CoordinateTracker...")
        tracker = hce.PyCoordinateTracker()
        tracker.add_message("msg_1", "system", "Test message", 10)
        stats = tracker.stats()
        print(f"  添加 1 条消息，总 token 数: {stats['total_tokens']}")
        
        # 测试 ContextCompressor
        print("\n🧪 测试 ContextCompressor...")
        compressor = hce.PyContextCompressor("keep_recent", 2)
        messages = json.dumps([
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User 1"},
            {"role": "assistant", "content": "Assistant 1"},
            {"role": "user", "content": "User 2"}
        ])
        result = compressor.compress(messages, [10, 5, 8, 5])
        print(f"  压缩比: {result['compression_ratio']:.1%}")
        
        print("\n✅ Context Engine 测试通过")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def cmd_test_router():
    """测试多模型路由器"""
    print("=" * 70)
    print("测试多模型路由器")
    print("=" * 70)
    
    try:
        import hermes_multi_model_router as hmr
        
        print("\n🧪 测试路由器创建...")
        router = hmr.PyRouter(None)
        print("  ✅ 路由器创建成功")
        
        print("\n🧪 测试并发模式...")
        modes = [
            ("single", None),
            ("top_n", 3),
            ("all", None),
        ]
        for mode, param in modes:
            router.set_concurrent_mode(mode, param)
            print(f"  ✅ 设置模式: {mode}")
        
        print("\n🧪 测试仲裁策略...")
        strategies = ["first_success", "fastest", "longest"]
        for strategy in strategies:
            router.set_arbitration_strategy(strategy)
            print(f"  ✅ 设置策略: {strategy}")
        
        print("\n✅ 多模型路由器测试通过")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def cmd_verify():
    """运行完整验证"""
    print("=" * 70)
    print("运行完整验证")
    print("=" * 70)
    
    # 运行所有测试
    cmd_test_context_engine()
    print()
    cmd_test_router()
    print()
    cmd_status()

def cmd_docs():
    """显示文档位置"""
    print("=" * 70)
    print("文档位置")
    print("=" * 70)
    
    docs = [
        ('快速参考', '~/.hermes/core-reform/QUICK_REFERENCE.md'),
        ('项目总结', '~/.hermes/core-reform/PROJECT_SUMMARY.md'),
        ('最终报告', '~/.hermes/core-reform/docs/final_complete_report.md'),
        ('集成报告', '~/.hermes/core-reform/docs/integration_complete_report.md'),
    ]
    
    print()
    for name, path in docs:
        full_path = os.path.expanduser(path)
        exists = "✅" if os.path.exists(full_path) else "❌"
        print(f"{exists} {name}:")
        print(f"   {path}")
        print()

def cmd_help():
    """显示帮助信息"""
    print("""
核心改造管理工具

用法: python core_reform_manager.py <command> [options]

命令:
  status              显示核心改造状态
  stats               显示详细轨迹统计
  export [path]       导出轨迹到文件
  test-context        测试 Context Engine
  test-router         测试多模型路由器
  verify              运行完整验证
  docs                显示文档位置
  help                显示此帮助信息

示例:
  python core_reform_manager.py status
  python core_reform_manager.py export /tmp/traces.json
  python core_reform_manager.py verify
    """)

def main():
    parser = argparse.ArgumentParser(description='核心改造管理工具')
    parser.add_argument('command', nargs='?', default='help',
                       choices=['status', 'stats', 'export', 'test-context', 
                               'test-router', 'verify', 'docs', 'help'])
    parser.add_argument('args', nargs='*', help='命令参数')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        cmd_status()
    elif args.command == 'stats':
        cmd_stats()
    elif args.command == 'export':
        output_path = args.args[0] if args.args else None
        cmd_export(output_path)
    elif args.command == 'test-context':
        cmd_test_context_engine()
    elif args.command == 'test-router':
        cmd_test_router()
    elif args.command == 'verify':
        cmd_verify()
    elif args.command == 'docs':
        cmd_docs()
    else:
        cmd_help()

if __name__ == '__main__':
    main()
