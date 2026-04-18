#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$HOME/Documents/hermes-agent"
MAIN_BRANCH="main"
DEV_BRANCH="dev"

cd "$REPO_DIR"

echo "==> 当前仓库: $REPO_DIR"
echo "==> 当前分支: $(git branch --show-current)"

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "错误：未配置 origin。"
  exit 1
fi

if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "错误：未配置 upstream，请先执行："
  echo "git remote add upstream https://github.com/NousResearch/hermes-agent.git"
  exit 1
fi

if [ ! -d "$REPO_DIR/venv" ]; then
  echo "错误：未找到 venv 目录，请先完成首次安装。"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "错误：当前工作区有未提交改动，请先提交或 stash。"
  exit 1
fi

echo "==> 获取 upstream 最新代码"
git fetch upstream

echo "==> 获取 origin 最新代码"
git fetch origin

echo "==> 切换到 $MAIN_BRANCH"
git checkout "$MAIN_BRANCH"

echo "==> 确保本地 $MAIN_BRANCH 跟踪 origin/$MAIN_BRANCH"
git branch --set-upstream-to="origin/$MAIN_BRANCH" "$MAIN_BRANCH" >/dev/null 2>&1 || true

echo "==> 合并 upstream/main 到 $MAIN_BRANCH"
git merge upstream/main

echo "==> 推送 $MAIN_BRANCH 到 origin"
git push origin "$MAIN_BRANCH"

echo "==> 切换到 $DEV_BRANCH"
git checkout "$DEV_BRANCH"

echo "==> 确保本地 $DEV_BRANCH 跟踪 origin/$DEV_BRANCH"
git branch --set-upstream-to="origin/$DEV_BRANCH" "$DEV_BRANCH" >/dev/null 2>&1 || true

echo "==> 合并 $MAIN_BRANCH 到 $DEV_BRANCH"
git merge "$MAIN_BRANCH"

echo "==> 推送 $DEV_BRANCH 到 origin"
git push origin "$DEV_BRANCH"

echo "==> 更新 submodules"
git submodule update --init --recursive

echo "==> 刷新 editable install"
export VIRTUAL_ENV="$REPO_DIR/venv"
uv pip install -e ".[all]"

if [ -d "$REPO_DIR/tinker-atropos" ]; then
  echo "==> 刷新 tinker-atropos"
  uv pip install -e "./tinker-atropos"
fi

echo "==> 检查配置"
hermes config check || true

echo "==> 迁移配置"
hermes config migrate || true

echo "==> 健康检查"
hermes doctor || true

echo "==> 版本信息"
hermes version || true

echo "==> 完成"
echo "当前分支: $(git branch --show-current)"
git status --short