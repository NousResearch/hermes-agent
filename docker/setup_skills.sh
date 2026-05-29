#!/bin/bash

set -e

export PATH="/opt/hermes/.venv/bin:${PATH}"

echo "========================================="
echo "Agent Toolkit Setup Script"
echo "========================================="

# 检查必需的环境变量
if [ -z "$HERMES_HOME" ]; then
    echo "错误: 环境变量 HERMES_HOME 未设置"
    exit 1
fi

if [ -z "$AGENT_TOOLKIT_SERVER" ]; then
    echo "错误: 环境变量 AGENT_TOOLKIT_SERVER 未设置"
    exit 1
fi

if [ -z "$LOOP_AGENT_CENTER_URL" ]; then
    echo "错误: 环境变量 LOOP_AGENT_CENTER_URL 未设置"
    exit 1
fi

if [ -z "$HERMES_WORKSPACE" ]; then
    echo "错误: 环境变量 HERMES_WORKSPACE 未设置"
    exit 1
fi

# 确保工作空间目录存在
if [ ! -d "$HERMES_WORKSPACE" ]; then
    echo "创建工作空间目录: $HERMES_WORKSPACE"
    mkdir -p "$HERMES_WORKSPACE"
    echo "✓ 工作空间目录创建成功"
fi

CONFIG_FILE="$HERMES_HOME/config.yaml"
SKILLS_DIR="$HERMES_HOME/skills"
SOUL_FILE="$HERMES_HOME/SOUL.md"
MEMORIES_DIR="$HERMES_HOME/memories"
MEMORY_FILE="$MEMORIES_DIR/MEMORY.md"

echo "HERMES_HOME: $HERMES_HOME"
echo "AGENT_TOOLKIT_SERVER: $AGENT_TOOLKIT_SERVER"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "SKILLS_DIR: $SKILLS_DIR"
echo "SOUL_FILE: $SOUL_FILE"
echo "MEMORY_FILE: $MEMORY_FILE"
echo ""

# agent-browser: 建立浏览器路径软链接
_AB_SRC="/opt/hermes/.agent-browser"
_AB_DST="$HERMES_HOME/.agent-browser"
if [ -d "$_AB_SRC/browsers" ]; then
    echo "发现 agent-browser 浏览器: $_AB_SRC"
    rm -rf "$_AB_DST"
    ln -s "$_AB_SRC" "$_AB_DST"
    echo "✓ agent-browser 浏览器路径已连接: $_AB_DST → $_AB_SRC"
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 功能1: 下载并解压skills
echo "========================================="
echo "功能1: 下载并解压Skills"
echo "========================================="

# 创建skills目录
mkdir -p "$SKILLS_DIR"

# 使用Python解析YAML并提取skills
SKILLS_JSON=$(python3 -c "
import yaml
import sys
import json

try:
    with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    skills = config.get('skill_templates', [])
    if not skills:
        print('[]', file=sys.stderr)
        sys.exit(0)

    # 转换为JSON格式
    print(json.dumps(skills))
except Exception as e:
    print(f'解析配置文件失败: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

if [ $? -ne 0 ]; then
    echo "错误: 解析配置文件失败"
    echo "$SKILLS_JSON"
    exit 1
fi

# 检查是否有skills需要下载
SKILLS_COUNT=$(echo "$SKILLS_JSON" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data) if data else 0)")

if [ "$SKILLS_COUNT" -eq 0 ]; then
    echo "配置文件中没有skills需要下载"
else
    echo "发现 $SKILLS_COUNT 个skills需要下载"

    # 遍历每个skill
    for i in $(seq 0 $((SKILLS_COUNT - 1))); do
        SKILL_INFO=$(echo "$SKILLS_JSON" | python3 -c "import sys, json; data = json.load(sys.stdin); print(json.dumps(data[$i]))")
        SKILL_ID=$(echo "$SKILL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")
        SKILL_VERSION=$(echo "$SKILL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('version', ''))")

        if [ -z "$SKILL_ID" ] || [ -z "$SKILL_VERSION" ]; then
            echo "警告: 跳过无效的skill配置 (id=$SKILL_ID, version=$SKILL_VERSION)"
            continue
        fi

        echo ""
        echo "正在下载 skill: id=$SKILL_ID, version=$SKILL_VERSION"

        SKILL_DIR_NAME="$SKILL_ID"
        TARGET_DIR="$SKILLS_DIR/$SKILL_DIR_NAME"

        # 如果已存在则跳过（Agent 运行时可能修改了 skill）
        if [ -d "$TARGET_DIR" ]; then
            echo "  skill 已存在，跳过: $TARGET_DIR"
            continue
        fi

        # 构造下载URL
        DOWNLOAD_URL="${AGENT_TOOLKIT_SERVER}/agent-toolkit/openapi/skills/${SKILL_ID}/download?version=${SKILL_VERSION}"

        # 创建临时目录
        TEMP_DIR=$(mktemp -d)
        TEMP_ZIP="$TEMP_DIR/skill.zip"

        # 下载skill
        echo "  下载URL: $DOWNLOAD_URL"
        HTTP_CODE=$(curl -s -o "$TEMP_ZIP" -w "%{http_code}" "$DOWNLOAD_URL")

        if [ "$HTTP_CODE" != "200" ]; then
            echo "  错误: 下载失败, HTTP状态码: $HTTP_CODE"
            rm -rf "$TEMP_DIR"
            continue
        fi

        # 检查下载的文件大小
        if [ ! -s "$TEMP_ZIP" ]; then
            echo "  错误: 下载的文件为空"
            rm -rf "$TEMP_DIR"
            continue
        fi

        # 检查文件是否是有效的zip文件（通过检查魔术数字 PK）
        if ! head -c 2 "$TEMP_ZIP" | grep -q "PK"; then
            echo "  错误: 下载的文件不是有效的zip文件"
            rm -rf "$TEMP_DIR"
            continue
        fi

        # 解压到临时目录
        echo "  解压skill包..."
        unzip -q -o "$TEMP_ZIP" -d "$TEMP_DIR"

        # 查找解压后的目录
        EXTRACTED_DIR=$(find "$TEMP_DIR" -maxdepth 1 -mindepth 1 -type d | head -n 1)

        if [ -z "$EXTRACTED_DIR" ]; then
            echo "  错误: 解压后未找到目录"
            rm -rf "$TEMP_DIR"
            continue
        fi

        # 移动到目标目录
        echo "  安装到: $TARGET_DIR"
        rm -rf "$TARGET_DIR"
        mv "$EXTRACTED_DIR" "$TARGET_DIR"

        # 清理临时目录
        rm -rf "$TEMP_DIR"

        echo "  ✓ skill安装成功"
    done
fi

echo ""

# 功能2: 生成soul.md文件
echo "========================================="
echo "功能2: 生成soul.md文件"
echo "========================================="

# 使用Python解析YAML并提取soul
SOUL_CONTENT=$(python3 -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    soul = config.get('soul', '')
    if soul is None:
        soul = ''

    print(soul)
except Exception as e:
    print(f'解析配置文件失败: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

if [ $? -ne 0 ]; then
    echo "错误: 解析soul配置失败"
    echo "$SOUL_CONTENT"
    exit 1
fi

# 检查soul内容是否为空
if [ -z "$SOUL_CONTENT" ]; then
    echo "配置文件中没有soul字段或soul为空"
    echo "跳过生成soul.md文件"
else
    echo "发现soul内容，正在生成 $SOUL_FILE ..."

    # 写入soul.md文件
    echo "$SOUL_CONTENT" > "$SOUL_FILE"

    echo "✓ soul.md文件生成成功"
fi

echo ""

# 功能3: 生成memory.md文件（环境事实，仅首次创建）
echo "========================================="
echo "功能3: 生成memory.md文件"
echo "========================================="

mkdir -p "$MEMORIES_DIR"

if [ -f "$MEMORY_FILE" ]; then
    echo "memory.md 已存在，跳过（保留 Agent 运行数据）"
else
    # 优先从 config.yaml 读取 memory 字段，fallback 到默认内容
    MEMORY_CONTENT=$(python3 -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    memory = config.get('memory', '')
    if memory is None:
        memory = ''

    if memory:
        print(memory, end='')
    else:
        # 默认环境事实
        print(\"\"\"# 环境事实

## 浏览器
- 当前环境已安装 Chrome 浏览器
- 所有浏览器操作请使用 agent-browser 工具
- 不要尝试其他浏览器或 Web 自动化方式

## 网络
- 运行在中国大陆
- 仅访问中国大陆可达的网站
- 优先使用国内镜像和 CDN 源\"\"\")
except Exception as e:
    print(f'解析配置文件失败: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

    if [ $? -ne 0 ]; then
        echo "错误: 解析memory配置失败"
        echo "$MEMORY_CONTENT"
        exit 1
    fi

    echo "$MEMORY_CONTENT" > "$MEMORY_FILE"
    echo "✓ memory.md文件生成成功"
fi

echo ""
echo "========================================="
echo "脚本执行完成"
echo "========================================="
