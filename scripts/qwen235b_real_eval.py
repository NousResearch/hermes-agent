#!/usr/bin/env python3
"""
Qwen235B 真实巡检场景评估脚本

评估流程：
1. Qwen235B模型生成响应
2. opencode作为第三方评估模型的响应
3. 评分基于客观标准

重点：不打开上帝视角，让模型根据获取的信息逐步分析
"""

import json
import os
import sys
import io
from pathlib import Path
from datetime import datetime

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ["HERMES_HOME"] = str(project_root / ".hermes")
os.environ["TERMINAL_CWD"] = str(project_root)

# 导入API调用器
from scripts.api_caller import call_qwen_api_via_powershell

def load_mes_inspection_skills():
    """加载MES巡检相关的skill"""
    skills_dir = project_root / "mes-inspection" / "skills"
    skills = {}
    
    for skill_dir in skills_dir.iterdir():
        if skill_dir.is_dir():
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                with open(skill_md, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    name = ""
                    description = ""
                    for line in lines[:20]:
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip()
                        elif line.startswith("description:"):
                            description = line.split(":", 1)[1].strip()
                    
                    if name:
                        skills[name] = {
                            "description": description,
                            "content": content
                        }
    
    return skills

def create_system_prompt(skills):
    """创建系统提示词，包含MES巡检skill信息"""
    prompt = """你是一位MES系统专家，擅长分析系统巡检数据并诊断问题。

你拥有以下MES巡检技能：
"""
    
    for skill_name, skill_info in skills.items():
        prompt += f"- **{skill_name}**: {skill_info['description']}\n"
    
    prompt += """
当收到巡检异常报告时，你应该：
1. 首先识别异常指标
2. 使用相关的巡检技能获取更多信息
3. 分析可能的原因
4. 如果需要更多信息，明确指出需要调用哪个skill
5. 给出排查建议和解决方案

请确保你的推理过程清晰、逻辑严密。
如果信息不足，请明确指出需要获取什么信息，不要猜测或编造。
"""
    
    return prompt

def call_qwen235b(api_key, system_prompt, user_message, conversation_history=None):
    """调用Qwen235B模型"""
    messages = []
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    result = call_qwen_api_via_powershell(
        api_key=api_key,
        model="Qwen3-235B-A22B-w8a8",
        messages=messages,
        max_tokens=2000,
        temperature=0.7
    )
    
    if result["success"]:
        response_data = result["data"]
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content
        else:
            return f"API响应格式错误: {response_data}"
    else:
        return f"API调用失败: {result['error']}"

def evaluate_by_mimo(api_key, system_prompt, user_message, model_response, turn_number, scenario):
    """
    由mimoV2.5-pro作为第三方评估Qwen235B的响应
    """
    eval_prompt = f"""你是一个AI模型评估专家。请评估以下Qwen235B模型的响应质量。

## 评估场景
{scenario}

## 用户消息
{user_message}

## Qwen235B模型响应
{model_response}

## 评估标准（每项0-25分，总分100分）

1. **诊断准确性（0-25分）**：是否正确识别了问题的关键点
2. **推理过程质量（0-25分）**：是否有清晰的逻辑和因果分析
3. **解决方案建议（0-25分）**：是否提出了具体可操作的解决方案
4. **请求更多信息（0-25分）**：是否主动指出需要获取什么信息或调用哪个skill

请严格按照以下JSON格式返回评估结果，不要添加任何其他内容：
```json
{{
  "diagnosis_score": 分数,
  "reasoning_score": 分数,
  "solution_score": 分数,
  "info_score": 分数,
  "total_score": 总分,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "suggestions": ["建议1", "建议2"]
}}
```"""

    messages = [
        {"role": "system", "content": "你是一个AI模型评估专家，负责评估其他AI模型的响应质量。请严格按照要求的JSON格式返回评估结果。"},
        {"role": "user", "content": eval_prompt}
    ]
    
    result = call_qwen_api_via_powershell(
        api_key=api_key,
        model="Qwen3-235B-A22B-w8a8",  # 使用同一个API，但角色是评估者
        messages=messages,
        max_tokens=1000,
        temperature=0.3  # 低温度，确保评估稳定
    )
    
    if result["success"]:
        response_data = result["data"]
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # 尝试解析JSON
            try:
                # 提取JSON部分
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    eval_data = json.loads(json_str)
                    return eval_data
                else:
                    return {"error": "无法从响应中提取JSON", "raw_response": content}
            except json.JSONDecodeError as e:
                return {"error": f"JSON解析失败: {e}", "raw_response": content}
        else:
            return {"error": f"API响应格式错误: {response_data}"}
    else:
        return {"error": f"API调用失败: {result['error']}"}

def main():
    """主评估函数"""
    print("=== Qwen235B 真实巡检场景评估（第三方评估版）===\n")
    
    # 加载MES巡检skill
    print("1. 加载MES巡检skill...")
    skills = load_mes_inspection_skills()
    print(f"   加载了 {len(skills)} 个skill")
    for skill_name in skills.keys():
        print(f"   - {skill_name}")
    
    # 创建系统提示词
    print("\n2. 创建系统提示词...")
    system_prompt = create_system_prompt(skills)
    print(f"   系统提示词长度: {len(system_prompt)} 字符")
    
    # API密钥
    api_key = "sk-dooFBpzVWgrvf32YLPFfq5r63dEYHELlUjMT84KrEH5wG0zN"
    
    # 准备对话历史
    conversation_history = []
    
    # 存储所有轮次的响应和评估
    all_results = []
    
    # ============================================================
    # 第1轮：定时巡检触发
    # ============================================================
    print("\n=== 第1轮：定时巡检触发 ===")
    print("场景：定时巡检脚本发现异常，输出门控信号")
    
    user_message_1 = """===INSPECTION_REPORT===
{"component":"oracle","status":"CRITICAL","checks":[{"name":"session_usage","status":"CRITICAL","value":290.0,"threshold":100.0}],"summary":"Oracle会话使用率290%，超过阈值100%"}
{"component":"jvm","status":"CRITICAL","checks":[{"name":"blocked_threads","status":"CRITICAL","value":1400,"threshold":500}],"summary":"JVM阻塞线程数1400，超过阈值500"}
===END===
{"wakeAgent": true}

请分析上述巡检报告，识别异常并给出排查建议。"""
    
    print(f"用户消息长度: {len(user_message_1)} 字符")
    print("调用Qwen235B模型...")
    
    response_1 = call_qwen235b(api_key, system_prompt, user_message_1)
    print(f"\n模型响应长度: {len(response_1)} 字符")
    print(f"模型响应前500字符:\n{response_1[:500]}...")
    
    # 由mimoV2.5-pro评估
    eval_1 = evaluate_by_mimo(api_key, system_prompt, user_message_1, response_1, 1, "定时巡检触发")
    print(f"\n第1轮评估得分: {eval_1.get('total_score', 0)}/100")
    print(f"  诊断准确性: {eval_1.get('diagnosis_score', 0)}/25")
    print(f"  推理过程质量: {eval_1.get('reasoning_score', 0)}/25")
    print(f"  解决方案建议: {eval_1.get('solution_score', 0)}/25")
    print(f"  请求更多信息: {eval_1.get('info_score', 0)}/25")
    if 'strengths' in eval_1:
        print(f"  优点: {', '.join(eval_1['strengths'])}")
    if 'weaknesses' in eval_1:
        print(f"  不足: {', '.join(eval_1['weaknesses'])}")
    
    all_results.append({
        "turn": 1,
        "scenario": "定时巡检触发",
        "user_message": user_message_1,
        "model_response": response_1,
        "evaluation": eval_1
    })
    
    # 更新对话历史
    conversation_history.append({"role": "user", "content": user_message_1})
    conversation_history.append({"role": "assistant", "content": response_1})
    
    # ============================================================
    # 第2轮：Agent调用skill获取更多信息
    # ============================================================
    print("\n=== 第2轮：Agent调用skill获取更多信息 ===")
    print("场景：根据第1轮分析，Agent调用mes-jvm-check获取线程堆栈")
    
    user_message_2 = """我调用了mes-jvm-check技能，获取到以下线程堆栈信息：

{
  "service": "jvm",
  "timestamp": "2026-05-28T11:00:00Z",
  "status": "critical",
  "checks": {
    "threads": {
      "status": "critical",
      "total": 1450,
      "blocked": 1400,
      "waiting": 50,
      "deadlock_detected": false
    }
  },
  "thread_stacks": [
    {
      "thread_name": "HTTP-8080-exec-470",
      "thread_id": 92238,
      "state": "BLOCKED",
      "stack_trace": [
        "java.net.InetAddress.getLocalHost(InetAddress.java:1486)",
        "- waiting to lock <0x0000000242a1c118> (a java.lang.Object)",
        "oracle.jdbc.driver.T4CTTIoauthenticate.setSessionFields(T4CTTIoauthenticate.java:985)",
        "oracle.jdbc.driver.T4CTTIoauthenticate.<init>(T4CTTIoauthenticate.java:261)",
        "oracle.jdbc.driver.T4CConnection.logon(T4CConnection.java:565)",
        "oracle.jdbc.driver.PhysicalConnection.<init>(PhysicalConnection.java:715)"
      ]
    }
  ]
}

请分析这些线程堆栈信息，识别问题的根本原因。"""
    
    print(f"用户消息长度: {len(user_message_2)} 字符")
    print("调用Qwen235B模型...")
    
    response_2 = call_qwen235b(api_key, system_prompt, user_message_2, conversation_history)
    print(f"\n模型响应长度: {len(response_2)} 字符")
    print(f"模型响应前500字符:\n{response_2[:500]}...")
    
    # 由mimoV2.5-pro评估
    eval_2 = evaluate_by_mimo(api_key, system_prompt, user_message_2, response_2, 2, "Agent调用skill获取更多信息")
    print(f"\n第2轮评估得分: {eval_2.get('total_score', 0)}/100")
    print(f"  诊断准确性: {eval_2.get('diagnosis_score', 0)}/25")
    print(f"  推理过程质量: {eval_2.get('reasoning_score', 0)}/25")
    print(f"  解决方案建议: {eval_2.get('solution_score', 0)}/25")
    print(f"  请求更多信息: {eval_2.get('info_score', 0)}/25")
    if 'strengths' in eval_2:
        print(f"  优点: {', '.join(eval_2['strengths'])}")
    if 'weaknesses' in eval_2:
        print(f"  不足: {', '.join(eval_2['weaknesses'])}")
    
    all_results.append({
        "turn": 2,
        "scenario": "Agent调用skill获取更多信息",
        "user_message": user_message_2,
        "model_response": response_2,
        "evaluation": eval_2
    })
    
    # 更新对话历史
    conversation_history.append({"role": "user", "content": user_message_2})
    conversation_history.append({"role": "assistant", "content": response_2})
    
    # ============================================================
    # 第3轮：Agent继续深入分析
    # ============================================================
    print("\n=== 第3轮：Agent继续深入分析 ===")
    print("场景：根据第2轮分析，Agent请求DNS排查")
    
    user_message_3 = """我执行了DNS排查命令，获取到以下结果：

# 检查/etc/hosts文件
$ cat /etc/hosts | grep $(hostname)
127.0.0.1   localhost

# 检查主机名解析
$ getent hosts $(hostname)
# 无输出

# 测试DNS解析速度
$ time nslookup $(hostname)
;; connection timed out; no servers could be reached

real    0m10.001s
user    0m0.001s
sys     0m0.001s

# 检查DNS配置
$ cat /etc/resolv.conf
nameserver 10.0.0.1
nameserver 10.0.0.2

# 测试DNS服务器连通性
$ ping -c 3 10.0.0.1
PING 10.0.0.1 (10.0.0.1) 56(84) bytes of data.
--- 10.0.0.1 ping statistics ---
3 packets transmitted, 0 received, 100% packet loss, time 2000ms

请分析这些DNS排查结果，确认是否是DNS问题导致的JDBC连接异常。"""
    
    print(f"用户消息长度: {len(user_message_3)} 字符")
    print("调用Qwen235B模型...")
    
    response_3 = call_qwen235b(api_key, system_prompt, user_message_3, conversation_history)
    print(f"\n模型响应长度: {len(response_3)} 字符")
    print(f"模型响应前500字符:\n{response_3[:500]}...")
    
    # 由mimoV2.5-pro评估
    eval_3 = evaluate_by_mimo(api_key, system_prompt, user_message_3, response_3, 3, "Agent继续深入分析")
    print(f"\n第3轮评估得分: {eval_3.get('total_score', 0)}/100")
    print(f"  诊断准确性: {eval_3.get('diagnosis_score', 0)}/25")
    print(f"  推理过程质量: {eval_3.get('reasoning_score', 0)}/25")
    print(f"  解决方案建议: {eval_3.get('solution_score', 0)}/25")
    print(f"  请求更多信息: {eval_3.get('info_score', 0)}/25")
    if 'strengths' in eval_3:
        print(f"  优点: {', '.join(eval_3['strengths'])}")
    if 'weaknesses' in eval_3:
        print(f"  不足: {', '.join(eval_3['weaknesses'])}")
    
    all_results.append({
        "turn": 3,
        "scenario": "Agent继续深入分析",
        "user_message": user_message_3,
        "model_response": response_3,
        "evaluation": eval_3
    })
    
    # 更新对话历史
    conversation_history.append({"role": "user", "content": user_message_3})
    conversation_history.append({"role": "assistant", "content": response_3})
    
    # ============================================================
    # 第4轮：Agent总结分析结果
    # ============================================================
    print("\n=== 第4轮：Agent总结分析结果 ===")
    print("场景：根据所有信息，Agent总结分析结果")
    
    user_message_4 = """请根据以上所有信息，总结分析结果，给出：
1. 问题的根本原因
2. 影响范围评估
3. 解决方案建议
4. 预防措施"""
    
    print(f"用户消息长度: {len(user_message_4)} 字符")
    print("调用Qwen235B模型...")
    
    response_4 = call_qwen235b(api_key, system_prompt, user_message_4, conversation_history)
    print(f"\n模型响应长度: {len(response_4)} 字符")
    print(f"模型响应前500字符:\n{response_4[:500]}...")
    
    # 由mimoV2.5-pro评估
    eval_4 = evaluate_by_mimo(api_key, system_prompt, user_message_4, response_4, 4, "Agent总结分析结果")
    print(f"\n第4轮评估得分: {eval_4.get('total_score', 0)}/100")
    print(f"  诊断准确性: {eval_4.get('diagnosis_score', 0)}/25")
    print(f"  推理过程质量: {eval_4.get('reasoning_score', 0)}/25")
    print(f"  解决方案建议: {eval_4.get('solution_score', 0)}/25")
    print(f"  请求更多信息: {eval_4.get('info_score', 0)}/25")
    if 'strengths' in eval_4:
        print(f"  优点: {', '.join(eval_4['strengths'])}")
    if 'weaknesses' in eval_4:
        print(f"  不足: {', '.join(eval_4['weaknesses'])}")
    
    all_results.append({
        "turn": 4,
        "scenario": "Agent总结分析结果",
        "user_message": user_message_4,
        "model_response": response_4,
        "evaluation": eval_4
    })
    
    # 计算平均分
    valid_scores = [r["evaluation"].get("total_score", 0) for r in all_results if "total_score" in r["evaluation"]]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"\n=== 评估总结 ===")
    print(f"平均得分: {avg_score:.1f}/100")
    
    # 保存评估结果
    eval_result = {
        "timestamp": datetime.now().isoformat(),
        "evaluated_model": "Qwen3-235B-A22B-w8a8",
        "evaluator_model": "Qwen3-235B-A22B-w8a8 (作为mimoV2.5-pro的替代)",
        "api_endpoint": "https://ai-pool.evebattery.com/v1/chat/completions",
        "system_prompt": system_prompt,
        "scenario": "真实巡检场景模拟（不打开上帝视角）",
        "conversation_turns": all_results,
        "average_score": avg_score
    }
    
    # 保存到文件
    result_file = project_root / "scripts" / "qwen235b_real_eval_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存到: {result_file}")
    print("\n评估完成！")

if __name__ == "__main__":
    main()
