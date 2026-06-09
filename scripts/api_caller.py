#!/usr/bin/env python3
"""
API调用器
调用Qwen235B API进行多步推理
"""

import requests
import json
import time
import os
import subprocess
import sys


def safe_nested_get(data, *keys, default="未知"):
    """
    安全的嵌套字典取值
    
    Args:
        data (dict): 源数据
        *keys: 键路径
        default: 默认值
        
    Returns:
        取到的值或默认值
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current


def call_qwen_api_via_powershell(api_key, model, messages, max_tokens=2000, temperature=0.7):
    """
    使用PowerShell调用Qwen API（备用方法，用于解决Python SSL库版本问题）
    
    Args:
        api_key (str): API密钥
        model (str): 模型名称
        messages (list): 消息列表
        max_tokens (int): 最大token数
        temperature (float): 温度参数
        
    Returns:
        dict: 统一包装格式 {"success": bool, "data": ..., "error": ...}
    """
    url = "https://ai-pool.evebattery.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    import tempfile
    temp_file = None
    ps_script_file = None
    try:
        # 写入payload到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=True)
            temp_file = f.name
        
        # 构建PowerShell脚本（使用字符串拼接避免f-string大括号转义问题）
        ps_script = (
            "$headers = @{\n"
            '    "Content-Type" = "application/json"\n'
            '    "Authorization" = "Bearer ' + api_key + '"\n'
            "}\n"
            '$body = Get-Content -Path "' + temp_file + '" -Raw\n'
            "try {\n"
            '    $response = Invoke-WebRequest -Uri "' + url + '" -Method POST -Headers $headers -Body $body -TimeoutSec 120\n'
            "    Write-Output $response.Content\n"
            "} catch {\n"
            "    Write-Error $_.Exception.Message\n"
            "    exit 1\n"
            "}\n"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as f:
            f.write(ps_script)
            ps_script_file = f.name
        
        # 执行PowerShell脚本文件
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_script_file],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            return {"success": True, "data": response_data, "error": None}
        else:
            return {"success": False, "data": None, "error": f"PowerShell调用失败: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "data": None, "error": "PowerShell调用超时"}
    except json.JSONDecodeError as e:
        return {"success": False, "data": None, "error": f"JSON解析失败: {e}"}
    except Exception as e:
        return {"success": False, "data": None, "error": f"PowerShell调用异常: {str(e)}"}
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass
        if ps_script_file and os.path.exists(ps_script_file):
            try:
                os.unlink(ps_script_file)
            except:
                pass


def call_qwen_api(api_key, model, messages, max_tokens=2000, temperature=0.7):
    """
    调用Qwen API（优先使用requests库，失败时使用PowerShell备用方法）
    
    Args:
        api_key (str): API密钥
        model (str): 模型名称
        messages (list): 消息列表
        max_tokens (int): 最大token数
        temperature (float): 温度参数
        
    Returns:
        dict: 统一包装格式 {"success": bool, "data": ..., "error": ...}
    """
    url = "https://ai-pool.evebattery.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # 首先尝试使用requests库
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return {"success": True, "data": response.json(), "error": None}
    except requests.exceptions.RequestException as e:
        # requests库失败，使用PowerShell备用方法
        print(f"requests库调用失败: {e}")
        print("尝试使用PowerShell备用方法...")
        return call_qwen_api_via_powershell(api_key, model, messages, max_tokens, temperature)
    except ValueError as e:
        return {"success": False, "data": None, "error": f"JSON解析失败: {e}"}


def create_analysis_prompt(scenario_data):
    """
    创建分析提示词
    
    Args:
        scenario_data (dict): 事故场景数据
        
    Returns:
        list: 消息列表
    """
    # 构建巡检数据摘要
    inspection_summary = []
    
    # Oracle检查结果
    oracle = scenario_data.get("oracle", {})
    inspection_summary.append(f"Oracle数据库状态: {oracle.get('status', '未知')}")
    inspection_summary.append(f"活跃会话数: {safe_nested_get(oracle, 'checks', 'sessions', 'active')}")
    inspection_summary.append(f"会话使用率: {safe_nested_get(oracle, 'checks', 'sessions', 'usage_percent')}%")
    
    # JVM检查结果
    jvm = scenario_data.get("jvm", {})
    inspection_summary.append(f"JVM应用状态: {jvm.get('status', '未知')}")
    inspection_summary.append(f"总线程数: {safe_nested_get(jvm, 'checks', 'thread_dump', 'total_threads')}")
    inspection_summary.append(f"阻塞线程数: {safe_nested_get(jvm, 'checks', 'thread_dump', 'blocked_threads')}")
    
    # 网络检查结果
    network = scenario_data.get("network", {})
    inspection_summary.append(f"DNS解析状态: {safe_nested_get(network, 'checks', 'dns_resolution', 'status')}")
    inspection_summary.append(f"DNS解析超时次数: {safe_nested_get(network, 'checks', 'dns_resolution', 'timeout_count')}")
    inspection_summary.append(f"JDBC连接重置错误: {safe_nested_get(network, 'checks', 'connection_reset', 'count')}次")
    
    # 事故背景
    accident = scenario_data.get("accident_summary", {})
    inspection_summary.append(f"事故时间: {accident.get('time', '未知')}")
    inspection_summary.append(f"影响范围: {accident.get('scope', '未知')}")
    inspection_summary.append(f"触发条件: {accident.get('trigger', '未知')}")
    inspection_summary.append(f"历史记录: {accident.get('history', '未知')}")
    
    inspection_text = "\n".join(inspection_summary)
    
    # 构建提示词
    prompt = f"""你是一位MES系统专家，擅长分析系统巡检数据并诊断问题。请分析以下巡检数据，并按照以下步骤进行推理：

1. **识别异常指标**：从数据中找出异常的指标。
2. **分析可能的原因**：基于异常指标，分析可能导致问题的原因。
3. **确定根本原因**：从可能的原因中，确定最可能的根本原因。
4. **提出解决方案**：针对根本原因，提出具体的解决方案。

请确保你的推理过程清晰、逻辑严密，并提供详细的解释。

**巡检数据**：
{inspection_text}

**事故背景**：
{accident.get('history', '未知')}

请开始你的分析："""
    
    return [
        {"role": "system", "content": "你是一位MES系统专家，擅长分析系统巡检数据并诊断问题。"},
        {"role": "user", "content": prompt}
    ]


if __name__ == "__main__":
    # 测试API调用 - 从环境变量读取API密钥
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        print("错误：请设置环境变量 QWEN_API_KEY")
        print("示例：set QWEN_API_KEY=your_api_key_here")
        exit(1)
    
    model = "Qwen3-235B-A22B-w8a8"
    
    # 测试消息
    messages = [
        {"role": "system", "content": "你是一位MES系统专家。"},
        {"role": "user", "content": "请简要介绍一下MES系统中常见的数据库连接问题。"}
    ]
    
    result = call_qwen_api(api_key, model, messages, max_tokens=500)
    print(json.dumps(result, indent=2, ensure_ascii=False))
