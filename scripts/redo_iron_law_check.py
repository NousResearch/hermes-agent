#!/usr/bin/env python3
"""
ppt-redo 铁律执法脚本 v2 — 四道检查（经内部+外部三轮第十人审计修复）
用法: python3 redo_iron_law_check.py <project_dir> [--mod <修改意见路径>]

退出码: 0=PASS, 1=WARN (有发现但非阻断), 2=FAIL (阻断级违规)
"""

import sys, os, re, json
from pathlib import Path

# ── 工具函数 ──

def safe_read(path):
    """多编码兼容读取"""
    for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode {path} with any encoding")


def extract_design_params(source_text):
    """从 source.md 提取设计要求——返回 {key: value} 其中 value 为 HEX 颜色或描述"""
    params = {}
    # 配色: 优先提取 HEX 值，回退到完整描述
    for key, label in [('primary', '主色'), ('accent', '强调色'), ('text', '文字色'), ('secondary', '辅色')]:
        # 先尝试提取 HEX
        m = re.search(rf'{label}[：:].*?(#[0-9a-fA-F]{{6}})', source_text)
        if m:
            params[key] = m.group(1)
        else:
            # 回退：提取整行描述
            m = re.search(rf'{label}[：:]\s*(.+?)(?:$|\n)', source_text)
            if m:
                params[key] = m.group(1).strip()
    # 字体
    m = re.search(r'字体[：:]\s*(.+?)(?:$|\n)', source_text)
    if m:
        params['font'] = m.group(1).strip()
    # 密度
    m = re.search(r'密度[：:]\s*(.+?)(?:$|\n)', source_text)
    if m:
        params['density'] = m.group(1).strip()
    # 风格
    m = re.search(r'视觉风格[：:]\s*(.+?)(?:$|\n)', source_text)
    if m:
        params['style'] = m.group(1).strip()
    # 禁止项: 合并所有禁止行
    bans = re.findall(r'禁止[：:]\s*(.+?)(?:$|\n)', source_text)
    if bans:
        params['bans'] = '; '.join(bans)
    return params


def extract_page_prompts(text):
    """
    从文本中提取每页的配图描述，返回 {page_num: prompt}
    支持多种页面标记: ## P{N}, ## P{N} —, ## 第{N}页, ## page {N}
    不再静默吞异常
    """
    prompts = {}
    # 多种页面分页正则（按优先级）
    patterns = [
        (r'\n## P(\d+)\s', 'P{N}'),             # ## P3 标题
        (r'\n## 第(\d+)页', '第{N}页'),          # ## 第3页
        (r'\n## Page (\d+)', 'page {N}'),         # ## Page 3
    ]
    
    for pat, _ in patterns:
        pages = re.split(pat, text)
        if len(pages) >= 3:  # 至少匹配到一个页面
            for i in range(1, len(pages), 2):
                try:
                    page_num = int(pages[i])
                    content = pages[i+1] if i+1 < len(pages) else ''
                    # 配图标记: **配图**：, **配图：**, 配图：, **Image**:
                    for prompt_pat in [
                        r'\*\*配图[：:]\*\*\s*(.+?)(?=\n##|\n---|\n\*\*|\Z)',
                        r'配图[：:]\s*(.+?)(?=\n##|\n---|\Z)',
                        r'\*\*配图\*\*\s*(.+?)(?=\n##|\n---|\Z)',
                        r'\*\*Image\*\*[：:]\s*(.+?)(?=\n##|\n---|\Z)',
                    ]:
                        m = re.search(prompt_pat, content, re.DOTALL)
                        if m:
                            prompt = m.group(1).strip().lstrip('：:')  # 去掉前导冒号
                            if len(prompt) > 20:
                                prompts[page_num] = prompt
                            break
                except (ValueError, IndexError) as e:
                    print(f"  ⚠️ 页面解析异常 (pat={_}): {e}", file=sys.stderr)
            break  # 第一个匹配的模式即停止
    return prompts


def parse_image_prompts_md(img_text):
    """
    解析 PPT Master 的 image_prompts.md → {image_name: prompt}
    支持多种命名格式: img01_xxx, img_p1_xxx, cover_bg, p01_cover_hero
    """
    img_prompts = {}
    # 通用匹配: ### Image {N}: {name}.{ext}
    # name 可以是任意非空白字符
    for match in re.finditer(
        r'### Image \d+: (\S+?)\.\S+?\s*\n.*?\*\*Prompt\*\*:\s*\n\s*\n(.+?)(?=\n\*\*Alt|\n---|\n###|\Z)',
        img_text, re.DOTALL
    ):
        img_name = match.group(1)
        prompt = match.group(2).strip()
        if len(prompt) > 20:
            img_prompts[img_name] = prompt
    return img_prompts


def match_page_to_image(page_num, img_name):
    """判断图像是否属于指定页——支持 img{N}_xxx, img_p{N}_xxx, p{N}_xxx 格式"""
    # img05_xxx → page 5
    m = re.search(r'img[_\s]*p?(\d+)', img_name, re.IGNORECASE)
    if m:
        return int(m.group(1)) == page_num
    # p05_xxx → page 5
    m = re.search(r'^p(\d+)_', img_name, re.IGNORECASE)
    if m:
        return int(m.group(1)) == page_num
    return False


def prompt_is_detailed(prompt):
    """判断配图描述是否足够详细（≥50字符 + 含构图/风格/颜色关键词，支持中英文）"""
    cn_keywords = ['颜色', '配色', '风格', '构图', '背景', '前景', '渐变',
                   '矢量', '摄影', '信息图', '排版', '居中', '顶部', '左', '右']
    en_keywords = ['color', 'style', 'composition', 'background', 'foreground',
                   'gradient', 'vector', 'photography', 'layout', 'centered',
                   'infographic', 'top', 'left', 'right', 'minimalist']
    
    has_style = any(kw in prompt.lower() for kw in cn_keywords + en_keywords)
    has_color = bool(re.search(r'#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}', prompt))
    # 英文用词数判断，中文用字符数
    char_count = len(prompt)
    word_count = len(prompt.split())
    is_long = char_count >= 50 or word_count >= 30
    return is_long and (has_style or has_color)


def classify_issue(severity, msg):
    """返回 (severity, msg) 元组，severity: 'fatal'|'warn'|'info'"""
    return (severity, msg)


# ── 检查函数 ──

def check_iron_law_1(source_path, mod_path=None):
    """
    铁律 1: 修改意见至上
    """
    issues = []
    source_text = safe_read(source_path)
    
    # 阻断级危险词（明确表示修改意见被主动修改）
    fatal_words = ['已优化', '已简化', '概括为', '精简为', '调整为']
    # 警告级（可能在合法上下文出现）
    warn_words = ['省略', '略去']
    
    for dw in fatal_words:
        if dw in source_text:
            issues.append(classify_issue('fatal', f"铁律1: source.md 出现优化标记 '{dw}'——修改意见不可优化"))
    for dw in warn_words:
        if dw in source_text:
            issues.append(classify_issue('warn', f"铁律1: source.md 出现 '{dw}'——请确认非内容删减"))
    
    # 如果提供了修改意见路径，做实质性对比
    if mod_path and os.path.exists(mod_path):
        try:
            mod_text = safe_read(mod_path)
        except ValueError as e:
            issues.append(classify_issue('fatal', f"铁律1: 无法读取修改意见文件: {e}"))
            return issues
        
        mod_prompts = extract_page_prompts(mod_text)
        src_prompts = extract_page_prompts(source_text)
        
        if not mod_prompts:
            issues.append(classify_issue('warn', 
                "铁律1: --mod 文件格式与 source.md 分页格式不匹配，无法逐页对比配图描述。仅做 danger_words 检查。"
                "（如需逐页对比，修改意见需使用 '## P{N}' 或 '## 第{N}页' 分页标记）"))
            return issues
        
        for page, mod_prompt in mod_prompts.items():
            src_prompt = src_prompts.get(page, '')
            if not src_prompt:
                issues.append(classify_issue('warn', f"铁律1: P{page} 的配图描述在 source.md 中丢失"))
            elif len(mod_prompt) > 50 and len(src_prompt) < len(mod_prompt) * 0.7:
                reduction = 100 - len(src_prompt) * 100 // len(mod_prompt)
                issues.append(classify_issue('warn', 
                    f"铁律1: P{page} 配图描述从 {len(mod_prompt)} 字缩至 {len(src_prompt)} 字（减少 {reduction}%）"))
    
    if not issues:
        return [classify_issue('info', "铁律1 PASS")]
    return issues


def check_iron_law_2(source_path, spec_path):
    """
    铁律 2: source.md 全量注入 PPT Master
    """
    issues = []
    source_text = safe_read(source_path)
    spec_text = safe_read(spec_path) if os.path.exists(spec_path) else ''
    
    if not spec_text:
        issues.append(classify_issue('fatal', "铁律2: spec_lock.md 不存在"))
        return issues
    
    src_params = extract_design_params(source_text)
    
    if not src_params:
        issues.append(classify_issue('fatal', "铁律2: source.md 中未找到设计要求块（## 🎨 设计要求）"))
        return issues
    
    # 尝试结构化解析 spec_lock
    spec_colors = {}
    for match in re.finditer(r'(primary|accent|text|secondary|background)\s*[:=]\s*["\']?(#[0-9a-fA-F]{6})', spec_text):
        spec_colors[match.group(1)] = match.group(2)
    
    for key, val in src_params.items():
        if key == 'bans':
            # 禁止项：检查 spec_lock 和 image_prompts 是否包含相反指令
            ban_items = [b.strip() for b in val.split(';')]
            for ban in ban_items:
                # 简单的反向检查：如 "禁止粉红色" → 检查文本中是否出现 "pink"/"#FFC0CB"
                ban_lower = ban.lower()
                if '粉红' in ban_lower or '粉色' in ban_lower:
                    if re.search(r'pink|#ff[0-9a-f]{2}cb|#ffc0cb', spec_text, re.IGNORECASE):
                        issues.append(classify_issue('warn', f"铁律2: spec_lock 中出现禁止的粉红色系"))
                if '渐变' in ban_lower:
                    if 'gradient' in spec_text.lower():
                        issues.append(classify_issue('warn', f"铁律2: spec_lock 中出现禁止的 'gradient'"))
            continue
        
        # 如果提取到 HEX 值，优先在 spec_colors 中检查角色匹配
        if val.startswith('#') and key in spec_colors:
            if spec_colors[key] != val:
                issues.append(classify_issue('warn', 
                    f"铁律2: {key}色在 spec_lock 中为 {spec_colors[key]}（source.md 指定 {val}）"))
        elif val not in spec_text:
            issues.append(classify_issue('warn', f"铁律2: spec_lock.md 缺少 {key}={val}"))
    
    if not issues:
        param_str = ', '.join(f'{k}={v}' for k, v in src_params.items() if k != 'bans')
        return [classify_issue('info', f"铁律2 PASS: 设计参数全量注入 ({param_str})")]
    return issues


def check_iron_law_3(source_path, image_prompts_path):
    """
    铁律 3: 配图提示词不可篡改
    逐页匹配，仅检查足够详细的源 prompt 是否被 PPT Master 重写
    """
    issues = []
    source_text = safe_read(source_path)
    src_prompts = extract_page_prompts(source_text)
    
    img_prompts = {}
    if os.path.exists(image_prompts_path):
        img_text = safe_read(image_prompts_path)
        img_prompts = parse_image_prompts_md(img_text)
    else:
        issues.append(classify_issue('info', "铁律3: image_prompts.md 不存在，跳过"))
        return issues
    
    if not img_prompts:
        issues.append(classify_issue('warn', "铁律3: image_prompts.md 存在但未解析出任何 prompt——格式可能不兼容"))
        return issues
    
    # 逐页对照
    for page, src_prompt in src_prompts.items():
        if not prompt_is_detailed(src_prompt):
            continue  # 不够详细→允许扩展，由铁律4检查
        
        # 找该页对应的生成图像
        page_matched = False
        for img_name, gen_prompt in img_prompts.items():
            if not match_page_to_image(page, img_name):
                continue
            page_matched = True
            
            # 检查是否被显著重写（≥2倍扩展 = 篡改）
            if len(gen_prompt) > len(src_prompt) * 2:
                issues.append(classify_issue('fatal', 
                    f"铁律3: P{page} ({img_name}) 配图描述被 PPT Master 重写（{len(src_prompt)}→{len(gen_prompt)}字，≥2倍）"))
            elif len(gen_prompt) < len(src_prompt) * 0.5:
                issues.append(classify_issue('warn', 
                    f"铁律3: P{page} ({img_name}) 配图描述异常缩短（{len(src_prompt)}→{len(gen_prompt)}字）"))
            break
        
        if not page_matched and prompt_is_detailed(src_prompt):
            issues.append(classify_issue('warn', f"铁律3: P{page} 有详细配图描述但 image_prompts.md 中找不到对应图像"))
    
    if not any(sev in ('fatal', 'warn') for sev, _ in issues):
        detailed_pages = sum(1 for p in src_prompts.values() if prompt_is_detailed(p))
        return [classify_issue('info', f"铁律3 PASS: 配图提示词未被篡改 ({detailed_pages} 页详细描述原样使用)")]
    return issues


def check_iron_law_4(source_path, image_prompts_path):
    """
    铁律 4: 自动扩展必须用修改意见的风格约束
    """
    issues = []
    source_text = safe_read(source_path)
    src_params = extract_design_params(source_text)
    src_prompts = extract_page_prompts(source_text)
    
    if not os.path.exists(image_prompts_path):
        return issues
    
    img_text = safe_read(image_prompts_path)
    img_prompts = parse_image_prompts_md(img_text)
    
    # 检查 PPT Master 默认风格
    ppt_master_defaults = ['swiss-minimal', 'editorial-lite', 'default template', 'minimalist-swiss']
    for default_style in ppt_master_defaults:
        if default_style.lower() in img_text.lower():
            issues.append(classify_issue('fatal', 
                f"铁律4: image_prompts.md/spec 出现 PPT Master 默认风格 '{default_style}'——应使用修改意见的风格"))
    
    # 逐页检查需要扩展的 prompt 是否注入设计参数
    for page, src_prompt in src_prompts.items():
        if prompt_is_detailed(src_prompt):
            continue  # 铁律3覆盖
        
        # 找对应生成图像
        for img_name, gen_prompt in img_prompts.items():
            if not match_page_to_image(page, img_name):
                continue
            
            # 检查配色注入
            for key in ['primary', 'accent']:
                if key in src_params and src_params[key].startswith('#'):
                    hex_val = src_params[key]
                    if hex_val.lower() not in gen_prompt.lower():
                        issues.append(classify_issue('warn', 
                            f"铁律4: P{page} ({img_name}) 扩展 prompt 未使用 {key}={hex_val}"))
            
            # 检查禁止项
            if 'bans' in src_params:
                bans_lower = src_params['bans'].lower()
                gen_lower = gen_prompt.lower()
                if '粉红' in bans_lower and ('pink' in gen_lower or '#ffc0cb' in gen_lower):
                    issues.append(classify_issue('warn', f"铁律4: P{page} 扩展 prompt 使用禁止的粉红色系"))
                if '渐变' in bans_lower and 'gradient' in gen_lower:
                    issues.append(classify_issue('warn', f"铁律4: P{page} 扩展 prompt 使用禁止的 'gradient'"))
            break
    
    if not any(sev in ('fatal', 'warn') for sev, _ in issues):
        return [classify_issue('info', "铁律4 PASS: 自动扩展使用修改意见风格约束")]
    return issues


# ── 主逻辑 ──

def main():
    if len(sys.argv) < 2:
        print("用法: python3 redo_iron_law_check.py <project_dir> [--mod <修改意见路径>] [--source <source.md>] [--spec <spec_lock.md>] [--img-prompts <image_prompts.md>]")
        sys.exit(2)
    
    project_dir = sys.argv[1]
    mod_path = None
    source_override = None
    spec_override = None
    img_override = None
    
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--mod' and i + 1 < len(args):
            mod_path = args[i + 1]; i += 2
        elif args[i] == '--source' and i + 1 < len(args):
            source_override = args[i + 1]; i += 2
        elif args[i] == '--spec' and i + 1 < len(args):
            spec_override = args[i + 1]; i += 2
        elif args[i] == '--img-prompts' and i + 1 < len(args):
            img_override = args[i + 1]; i += 2
        else:
            i += 1
    
    # 路径查找
    source_path = source_override or os.path.join(project_dir, 'sources', 'source.md')
    if not os.path.exists(source_path):
        source_path = source_override or os.path.join(project_dir, 'source.md')
    if not os.path.exists(source_path) and not source_override:
        parent = os.path.dirname(os.path.abspath(project_dir))
        candidates = []
        for d in sorted(os.listdir(parent)):
            full = os.path.join(parent, d, 'sources', 'source.md')
            if os.path.exists(full):
                candidates.append(full)
        if len(candidates) == 1:
            source_path = candidates[0]
        elif len(candidates) > 1:
            print(f"⚠️ 发现多个 source.md，使用第一个: {candidates[0]}", file=sys.stderr)
            source_path = candidates[0]
    
    spec_path = spec_override or os.path.join(project_dir, 'spec_lock.md')
    img_prompts_path = img_override or os.path.join(project_dir, 'images', 'image_prompts.md')
    
    if not os.path.exists(source_path):
        print(f"❌ 找不到 source.md: 搜过 {project_dir}/sources/source.md, {project_dir}/source.md", file=sys.stderr)
        sys.exit(2)
    
    print(f"\n{'='*60}")
    print(f"📋 ppt-redo 铁律执法检查 v2")
    print(f"   source: {source_path}")
    print(f"   spec:   {spec_path} {'(不存在)' if not os.path.exists(spec_path) else ''}")
    print(f"   images: {img_prompts_path} {'(不存在)' if not os.path.exists(img_prompts_path) else ''}")
    print(f"{'='*60}\n")
    
    all_issues = []
    
    for label, check_fn, args in [
        ("铁律 1: 修改意见至上", check_iron_law_1, (source_path, mod_path)),
        ("铁律 2: source.md 全量注入", check_iron_law_2, (source_path, spec_path)),
        ("铁律 3: 配图提示词不可篡改", check_iron_law_3, (source_path, img_prompts_path)),
        ("铁律 4: 扩展用修改意见风格", check_iron_law_4, (source_path, img_prompts_path)),
    ]:
        print(f"🔍 {label}")
        try:
            issues = check_fn(*args)
        except Exception as e:
            issues = [classify_issue('fatal', f"{label}: 检查异常: {e}")]
        all_issues.extend(issues)
        for sev, msg in issues:
            prefix = {'fatal': '🔴', 'warn': '🟡', 'info': '  '}.get(sev, '  ')
            if sev != 'info':
                print(f"  {prefix} {msg}")
            else:
                print(f"  ✅ {msg}")
    
    # 总结（使用结构化 severity 字段）
    fatal = sum(1 for s, _ in all_issues if s == 'fatal')
    warn = sum(1 for s, _ in all_issues if s == 'warn')
    
    print(f"\n{'='*60}")
    if fatal > 0:
        print(f"❌ FAIL: {fatal} 阻断 + {warn} 警告")
        sys.exit(2)
    elif warn > 0:
        print(f"⚠️  WARN: {warn} 警告（非阻断）")
        sys.exit(1)
    else:
        print("✅ ALL PASS: 四条铁律全部通过")
        sys.exit(0)


if __name__ == '__main__':
    main()
