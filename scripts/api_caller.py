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
    调用Qwen API
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
    result_file = None
    try:
        # 写入payload到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=True)
            temp_file = f.name
        
        # 构建PowerShell脚本 - 使用StreamReader从响应流中读取UTF-8内容
        result_file = temp_file + ".result.json"
        ps_lines = []
        ps_lines.append('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8')
        ps_lines.append('$headers = @{')
        ps_lines.append('    "Content-Type" = "application/json; charset=utf-8"')
        ps_lines.append('    "Authorization" = "Bearer ' + api_key + '"')
        ps_lines.append('}')
        ps_lines.append('$body = [System.IO.File]::ReadAllText("' + temp_file.replace('\\', '\\\\') + '", [System.Text.Encoding]::UTF8)')
        ps_lines.append("try {")
        ps_lines.append('    $response = Invoke-WebRequest -Uri "' + url + '" -Method POST -Headers $headers -Body $body -TimeoutSec 120')
        ps_lines.append('    $stream = $response.RawContentStream')
        ps_lines.append('    $stream.Position = 0')
        ps_lines.append('    $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)')
        ps_lines.append('    $content = $reader.ReadToEnd()')
        ps_lines.append('    $reader.Close()')
        ps_lines.append('    $bytes = [System.Text.Encoding]::UTF8.GetBytes($content)')
        ps_lines.append('    [System.IO.File]::WriteAllBytes("' + result_file.replace('\\', '\\\\') + '", $bytes)')
        ps_lines.append("    exit 0")
        ps_lines.append("} catch {")
        ps_lines.append("    Write-Error $_.Exception.Message")
        ps_lines.append("    exit 1")
        ps_lines.append("}")
        ps_script = "\n".join(ps_lines)
        
        # 用UTF-8 BOM写入ps1脚本
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8-sig') as f:
            f.write(ps_script)
            ps_script_file = f.name
        
        # 执行PowerShell脚本文件
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_script_file],
            capture_output=True,
            timeout=180
        )
        
        if result.returncode == 0 and result_file and os.path.exists(result_file):
            # 从文件读取结果
            with open(result_file, 'rb') as f:
                raw_bytes = f.read()
            # 尝试多种编码解码
            for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
                try:
                    text = raw_bytes.decode(encoding)
                    response_data = json.loads(text)
                    return {"success": True, "data": response_data, "error": None}
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            # 所有编码都失败
            return {"success": False, "data": None, "error": f"无法解码响应内容"}
        else:
            stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else "未知错误"
            return {"success": False, "data": None, "error": f"PowerShell调用失败: {stderr_text}"}
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
