#!/usr/bin/env python3
"""测试轨迹钩子装饰器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.trace_hooks import trace_tool_call, get_trace_stats, export_traces

# 模拟工具调用函数
@trace_tool_call
def mock_tool_call(tool_name: str, tool_args: dict, task_id: str = None):
    """模拟的工具调用函数"""
    print(f"  执行工具: {tool_name}")
    print(f"  参数: {tool_args}")
    
    # 模拟不同的工具行为
    if tool_name == "error_tool":
        raise Exception("模拟错误")
    
    return f"工具 {tool_name} 执行成功"

print("🧪 测试轨迹钩子装饰器\n")

# 测试1：成功的工具调用
print("测试1：成功的工具调用")
result = mock_tool_call("test_tool", {"param1": "value1"}, "task_123")
print(f"  结果: {result}\n")

# 测试2：另一个成功的工具调用
print("测试2：另一个成功的工具调用")
result = mock_tool_call("another_tool", {"param2": "value2"}, "task_456")
print(f"  结果: {result}\n")

# 测试3：失败的工具调用
print("测试3：失败的工具调用")
try:
    result = mock_tool_call("error_tool", {"param3": "value3"}, "task_789")
except Exception as e:
    print(f"  捕获异常: {e}\n")

# 测试4：查看统计
print("测试4：查看统计")
stats = get_trace_stats()
if stats:
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  平均耗时: {stats['avg_duration_ms']:.2f}ms\n")

# 测试5：导出轨迹
print("测试5：导出轨迹")
export_path = "/tmp/test_traces.json"
export_traces(export_path)
print(f"  已导出到: {export_path}\n")

# 验证导出的文件
import json
with open(export_path, 'r') as f:
    traces = json.load(f)
    print(f"✅ 验证：导出了 {len(traces)} 条轨迹")
    if traces:
        print(f"  第一条轨迹: {traces[0]['tool_name']}")

print("\n✅ 所有测试通过！")
