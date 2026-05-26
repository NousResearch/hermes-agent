"""
Full Tool-Call Integration Module

将 Sidecar 系统集成到 Hermes 主链路，提供全局轨迹规划和执行能力。

核心功能：
1. 任务输入 → 全局轨迹规划
2. 并行任务调度
3. 负载均衡和故障自愈
4. 证据验证和状态追踪
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path.home() / '.hermes' / 'scripts'
PLANNER = SCRIPTS_DIR / 'hermes_full_toolcall_planner.py'
RUNTIME = SCRIPTS_DIR / 'hermes_full_toolcall_runtime.py'
WORKSPACE = Path.home() / '.hermes' / 'workspace' / '开智' / 'full_toolcall_sidecar'


@dataclass
class FullToolCallResult:
    """Full Tool-Call 执行结果"""
    success: bool
    plan_path: str
    run_path: str
    tasks_total: int
    completed: int
    failed: int
    events: int
    workers_used: List[str]
    healing_events: int
    error: Optional[str] = None


class FullToolCallIntegration:
    """Full Tool-Call 集成接口"""
    
    def __init__(self):
        """初始化集成接口"""
        self.planner = PLANNER
        self.runtime = RUNTIME
        self.workspace = WORKSPACE
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def plan(self, input_text: str, goal: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        生成全局轨迹规划
        
        Args:
            input_text: 输入材料（文本或文件路径）
            goal: 目标描述
            output_path: 输出路径（可选）
            
        Returns:
            规划结果
        """
        # 如果是文件路径，直接使用；否则写入临时文件
        if Path(input_text).exists():
            input_path = Path(input_text)
        else:
            input_path = self.workspace / 'temp_input.txt'
            input_path.write_text(input_text, encoding='utf-8')
        
        # 生成输出路径
        if output_path is None:
            output_path = self.workspace / 'plan_latest.json'
        
        # 调用规划器
        try:
            result = subprocess.run(
                [
                    'python3',
                    str(self.planner),
                    '--input', str(input_path),
                    '--goal', goal,
                    '--out', str(output_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"规划器执行失败: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            # 解析输出
            summary = json.loads(result.stdout)
            plan = json.loads(output_path.read_text(encoding='utf-8'))
            
            logger.info(f"✅ 规划完成: {summary['steps']} 步骤, {summary['tools']} 工具, {summary['risks']} 风险")
            
            return {
                'success': True,
                'plan_path': str(output_path),
                'plan': plan,
                'summary': summary
            }
            
        except subprocess.TimeoutExpired:
            logger.error("规划器执行超时")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.error(f"规划器执行异常: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute(self, plan_path: str, output_path: Optional[Path] = None, 
                simulate_failure: bool = False) -> FullToolCallResult:
        """
        执行全局轨迹
        
        Args:
            plan_path: 规划文件路径
            output_path: 输出路径（可选）
            simulate_failure: 是否模拟故障（用于测试）
            
        Returns:
            执行结果
        """
        # 生成输出路径
        if output_path is None:
            output_path = self.workspace / 'run_latest.json'
        
        # 调用运行时
        try:
            cmd = [
                'python3',
                str(self.runtime),
                '--plan', plan_path,
                '--out', str(output_path)
            ]
            
            if simulate_failure:
                cmd.append('--simulate-failure')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"运行时执行失败: {result.stderr}")
                return FullToolCallResult(
                    success=False,
                    plan_path=plan_path,
                    run_path=str(output_path),
                    tasks_total=0,
                    completed=0,
                    failed=0,
                    events=0,
                    workers_used=[],
                    healing_events=0,
                    error=result.stderr
                )
            
            # 解析输出
            summary = json.loads(result.stdout)
            run_data = json.loads(output_path.read_text(encoding='utf-8'))
            
            logger.info(f"✅ 执行完成: {summary['completed']}/{summary['tasks_total']} 任务, {summary['healing_events']} 次自愈")
            
            return FullToolCallResult(
                success=True,
                plan_path=plan_path,
                run_path=str(output_path),
                tasks_total=summary['tasks_total'],
                completed=summary['completed'],
                failed=summary['failed'],
                events=summary['events'],
                workers_used=summary['workers_used'],
                healing_events=summary['healing_events']
            )
            
        except subprocess.TimeoutExpired:
            logger.error("运行时执行超时")
            return FullToolCallResult(
                success=False,
                plan_path=plan_path,
                run_path=str(output_path),
                tasks_total=0,
                completed=0,
                failed=0,
                events=0,
                workers_used=[],
                healing_events=0,
                error='timeout'
            )
        except Exception as e:
            logger.error(f"运行时执行异常: {e}")
            return FullToolCallResult(
                success=False,
                plan_path=plan_path,
                run_path=str(output_path),
                tasks_total=0,
                completed=0,
                failed=0,
                events=0,
                workers_used=[],
                healing_events=0,
                error=str(e)
            )
    
    def plan_and_execute(self, input_text: str, goal: str, 
                        simulate_failure: bool = False) -> Dict[str, Any]:
        """
        规划并执行全局轨迹（一站式接口）
        
        Args:
            input_text: 输入材料
            goal: 目标描述
            simulate_failure: 是否模拟故障
            
        Returns:
            完整结果
        """
        # 1. 规划
        plan_result = self.plan(input_text, goal)
        if not plan_result['success']:
            return {
                'success': False,
                'stage': 'plan',
                'error': plan_result['error']
            }
        
        # 2. 执行
        exec_result = self.execute(
            plan_result['plan_path'],
            simulate_failure=simulate_failure
        )
        
        return {
            'success': exec_result.success,
            'stage': 'complete' if exec_result.success else 'execute',
            'plan': plan_result,
            'execution': exec_result,
            'summary': {
                'steps': plan_result['summary']['steps'],
                'completed': exec_result.completed,
                'failed': exec_result.failed,
                'success_rate': exec_result.completed / exec_result.tasks_total if exec_result.tasks_total > 0 else 0,
                'healing_events': exec_result.healing_events
            }
        }


# 便捷函数
def get_full_toolcall() -> FullToolCallIntegration:
    """获取 Full Tool-Call 集成实例"""
    return FullToolCallIntegration()


if __name__ == "__main__":
    # 测试
    import tempfile
    
    print("=" * 70)
    print("Full Tool-Call 集成测试")
    print("=" * 70)
    print()
    
    # 创建集成实例
    ftc = get_full_toolcall()
    
    # 测试输入
    test_input = """
    测试任务：验证 Full Tool-Call 集成
    
    步骤：
    1. 读取配置文件
    2. 执行测试脚本
    3. 验证结果
    4. 生成报告
    """
    
    test_goal = "验证 Full Tool-Call 集成功能"
    
    # 执行
    print("🚀 开始执行...")
    result = ftc.plan_and_execute(test_input, test_goal)
    
    print()
    print("📊 执行结果:")
    print(f"  成功: {result['success']}")
    print(f"  阶段: {result['stage']}")
    
    if result['success']:
        summary = result['summary']
        print(f"  步骤数: {summary['steps']}")
        print(f"  完成数: {summary['completed']}")
        print(f"  失败数: {summary['failed']}")
        print(f"  成功率: {summary['success_rate']:.2%}")
        print(f"  自愈次数: {summary['healing_events']}")
    else:
        print(f"  错误: {result.get('error', 'unknown')}")
    
    print()
    print("=" * 70)
    print("✅ Full Tool-Call 集成测试完成")
    print("=" * 70)
