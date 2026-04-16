#!/usr/bin/env python3
"""
测试 Lyapunov 监控与 harness-5 的完整集成
模拟真实场景：渐进式恶化与恢复
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_tools import _record_tool_failure

def simulate_progressive_degradation():
    """模拟渐进式系统恶化"""
    print("=== 模拟渐进式系统恶化 ===")
    
    session_id = "progressive_session"
    
    # 阶段 1: 健康状态
    print("\n1. 健康状态 (少量失败):")
    for i in range(3):
        result = _record_tool_failure(
            function_name=f"tool_{i}",
            error_message=f"Minor error {i}",
            session_id=session_id,
        )
        alarm_metadata = result.get('lyapunov_alarm', {})
        if isinstance(alarm_metadata, dict):
            v_value = alarm_metadata.get('metadata', {}).get('V', 'N/A')
        else:
            v_value = 'N/A'
        print(f"   失败 {i+1}: status={result.get('lyapunov_status', 'N/A')}, V={v_value}")
    
    # 阶段 2: 开始恶化（超时增加）
    print("\n2. 开始恶化 (超时增加):")
    for i in range(5):
        result = _record_tool_failure(
            function_name="terminal",
            error_message=f"Command timed out after {30+i*5} seconds",
            session_id=session_id,
        )
        status = result.get('lyapunov_status', 'N/A')
        alarm = result.get('lyapunov_alarm')
        print(f"   超时 {i+1}: status={status}, alarm={alarm is not None}")
    
    # 阶段 3: 严重恶化（上下文溢出）
    print("\n3. 严重恶化 (上下文溢出):")
    for i in range(3):
        result = _record_tool_failure(
            function_name="read_file",
            error_message="Context length exceeded maximum of 128000 tokens",
            session_id=session_id,
        )
        status = result.get('lyapunov_status', 'N/A')
        alarm = result.get('lyapunov_alarm')
        print(f"   溢出 {i+1}: status={status}, alarm_level={alarm.get('level') if alarm else 'NONE'}")
    
    # 阶段 4: 检测循环（最严重）
    print("\n4. 检测循环 (最严重):")
    for i in range(2):
        result = _record_tool_failure(
            function_name="execute_code",
            error_message=f"Loop detected: infinite recursion (count={i+3})",
            session_id=session_id,
        )
        status = result.get('lyapunov_status', 'N/A')
        alarm = result.get('lyapunov_alarm')
        print(f"   循环 {i+1}: status={status}, alarm_level={alarm.get('level') if alarm else 'NONE'}")
    
    print("\n=== 模拟完成 ===")
    
    # 获取最终状态
    print("\n=== 最终状态 ===")
    try:
        import sys
        from pathlib import Path
        hermes_home = Path.home() / ".hermes"
        scripts_path = str(hermes_home / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from lyapunov_health_monitor.integration import get_integrator
        
        integrator = get_integrator()
        status = integrator.get_combined_status(session_id)
        
        print(f"综合状态: {status.get('combined_status', 'N/A')}")
        print(f"Lyapunov 状态: {status.get('lyapunov', {}).get('status', 'N/A')}")
        print(f"能量值 V: {status.get('lyapunov', {}).get('V', 'N/A'):.2f}")
        print(f"能量变化 V̇: {status.get('lyapunov', {}).get('V_dot', 'N/A'):.2f}")
        print(f"警告数量: {len(status.get('lyapunov', {}).get('warnings', []))}")
        print(f"建议行动: {status.get('lyapunov', {}).get('actions', [])}")
        print(f"Harness 模式数量: {status.get('harness', {}).get('count', 0)}")
        
    except Exception as e:
        print(f"获取状态失败: {e}")

def test_preventive_actions():
    """测试预防性行动建议"""
    print("\n=== 测试预防性行动建议 ===")
    
    try:
        import sys
        from pathlib import Path
        hermes_home = Path.home() / ".hermes"
        scripts_path = str(hermes_home / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from lyapunov_health_monitor.integration import get_integrator
        
        integrator = get_integrator()
        
        # 测试不同 session 的预防性行动
        sessions = ["healthy_session", "warning_session", "critical_session"]
        
        for session in sessions:
            actions = integrator.get_preventive_actions(session)
            print(f"\n{session}:")
            if actions:
                for i, action in enumerate(actions, 1):
                    print(f"  {i}. {action}")
            else:
                print("  无需预防性行动")
                
    except Exception as e:
        print(f"测试预防性行动失败: {e}")

def test_integration_api():
    """测试集成 API"""
    print("\n=== 测试集成 API ===")
    
    try:
        import sys
        from pathlib import Path
        hermes_home = Path.home() / ".hermes"
        scripts_path = str(hermes_home / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from lyapunov_health_monitor.integration import (
            get_integrator,
            record_failure_with_monitoring,
        )
        
        print("1. 测试 record_failure_with_monitoring:")
        result = record_failure_with_monitoring(
            failure_type="integration_test",
            message="Testing integration API",
            context={"test": True, "session_id": "api_test"},
            session_id="api_test_session",
        )
        print(f"   Result: {result.get('alarm_level', 'N/A')}")
        
        print("\n2. 测试触发预防性压缩:")
        integrator = get_integrator()
        triggered = integrator.trigger_preventive_compression("progressive_session")
        print(f"   触发压缩: {triggered}")
        
        print("\n3. 测试启用/禁用监控:")
        integrator.disable()
        print("   监控已禁用")
        
        # 测试禁用后的监控
        result = _record_tool_failure(
            function_name="test",
            error_message="Test with monitoring disabled",
            session_id="disabled_test",
        )
        print(f"   禁用后状态: {result.get('lyapunov_status', 'N/A')}")
        
        integrator.enable()
        print("   监控已启用")
        
    except Exception as e:
        print(f"测试集成 API 失败: {e}")

if __name__ == "__main__":
    simulate_progressive_degradation()
    test_preventive_actions()
    test_integration_api()