#!/usr/bin/env python3
"""
测试 Lyapunov 监控与 model_tools.py 的集成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_tools import _record_tool_failure

def test_lyapunov_integration():
    """测试 Lyapunov 监控集成"""
    print("=== 测试 Lyapunov 监控集成 ===")
    
    # 测试 1: 工具超时
    print("\n1. 测试工具超时:")
    result = _record_tool_failure(
        function_name="terminal",
        error_message="Command timed out after 30 seconds",
        session_id="test_session_1",
        task_id="test_task_1",
    )
    print(f"   Result: {result}")
    print(f"   Lyapunov status: {result.get('lyapunov_status', 'N/A')}")
    print(f"   Lyapunov alarm: {result.get('lyapunov_alarm', 'N/A')}")
    
    # 测试 2: 上下文溢出
    print("\n2. 测试上下文溢出:")
    result = _record_tool_failure(
        function_name="read_file",
        error_message="Context length exceeded maximum of 128000 tokens",
        session_id="test_session_2",
        task_id="test_task_2",
    )
    print(f"   Result: {result}")
    print(f"   Lyapunov status: {result.get('lyapunov_status', 'N/A')}")
    
    # 测试 3: sudo 权限问题
    print("\n3. 测试 sudo 权限问题:")
    result = _record_tool_failure(
        function_name="terminal",
        error_message="Permission denied: sudo command requires user confirmation",
        session_id="test_session_3",
        task_id="test_task_3",
    )
    print(f"   Result: {result}")
    print(f"   Lyapunov status: {result.get('lyapunov_status', 'N/A')}")
    
    # 测试 4: 普通工具失败
    print("\n4. 测试普通工具失败:")
    result = _record_tool_failure(
        function_name="search_files",
        error_message="Pattern not found in any files",
        session_id="test_session_4",
        task_id="test_task_4",
    )
    print(f"   Result: {result}")
    print(f"   Lyapunov status: {result.get('lyapunov_status', 'N/A')}")
    
    print("\n=== 测试完成 ===")
    
    # 检查 Lyapunov 模块是否被正确加载
    print("\n=== 检查 Lyapunov 模块 ===")
    try:
        import sys
        from pathlib import Path
        hermes_home = Path.home() / ".hermes"
        scripts_path = str(hermes_home / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from lyapunov_health_monitor.integration import get_integrator
        
        integrator = get_integrator()
        print("✓ Lyapunov 集成器加载成功")
        
        # 获取综合状态
        status = integrator.get_combined_status("test_session_1")
        print(f"  综合状态: {status.get('combined_status', 'N/A')}")
        print(f"  Lyapunov V: {status.get('lyapunov', {}).get('V', 'N/A')}")
        print(f"  Harness patterns: {status.get('harness', {}).get('count', 0)}")
        
    except ImportError as e:
        print(f"✗ Lyapunov 模块导入失败: {e}")
    except Exception as e:
        print(f"✗ Lyapunov 模块错误: {e}")

if __name__ == "__main__":
    test_lyapunov_integration()