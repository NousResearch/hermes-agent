#!/usr/bin/env bash
# scripts/build-push-saas.sh — Build and push the Hermes SaaS Docker image.
#
# Usage:
#   ./scripts/build-push-saas.sh [TAG]
#
# Arguments:
#   TAG  — Docker image tag (default: plan-001-E)
#
# Environment:
#   AWS_PROFILE  — AWS CLI profile (default: AgenticHub-162471567408)
#   AWS_REGION   — AWS region (default: us-east-1)
#
# Example:
#   AWS_PROFILE=AgenticHub-162471567408 ./scripts/build-push-saas.sh plan-001-E
#
# This script:
#   1. Verifies AWS credentials (fails fast if not configured)
#   2. Logs in to ECR
#   3. Builds Dockerfile.saas with the given tag
#   4. Tags and pushes to ECR
#   5. Prints the full ECR image URI on success
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAG="${1:-plan-001-E}"
AWS_PROFILE="${AWS_PROFILE:-AgenticHub-162471567408}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="162471567408"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/agentic-stack/hermes"
FULL_URI="${ECR_REPO}:${TAG}"
LOCAL_IMAGE="hermes-saas:${TAG}"

# Resolve repo root (script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Build + push: ${FULL_URI}"
echo "    Repo root  : ${REPO_ROOT}"
echo "    AWS Profile: ${AWS_PROFILE}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Verify AWS credentials
# ---------------------------------------------------------------------------
echo "[1/5] Verifying AWS credentials..."
CALLER=$(AWS_PROFILE="${AWS_PROFILE}" aws sts get-caller-identity --region "${AWS_REGION}" --output json)
CALLER_ACCOUNT=$(echo "${CALLER}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Account'])")
CALLER_ARN=$(echo "${CALLER}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Arn'])")
echo "    Account: ${CALLER_ACCOUNT}"
echo "    Caller : ${CALLER_ARN}"

if [ "${CALLER_ACCOUNT}" != "${ACCOUNT_ID}" ]; then
    echo "ERROR: Expected account ${ACCOUNT_ID}, got ${CALLER_ACCOUNT}. Check AWS_PROFILE."
    exit 1
fi
echo "    OK — credentials valid."
echo ""

# ---------------------------------------------------------------------------
# Step 2: ECR login
# ---------------------------------------------------------------------------
echo "[2/5] Logging in to ECR..."
AWS_PROFILE="${AWS_PROFILE}" aws ecr get-login-password \
    --region "${AWS_REGION}" \
    | docker login \
        --username AWS \
        --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo "    OK — ECR login successful."
echo ""

# ---------------------------------------------------------------------------
# Step 3: Docker build
# ---------------------------------------------------------------------------
echo "[3/5] Building image: ${LOCAL_IMAGE}"
docker build \
    --file "${REPO_ROOT}/Dockerfile.saas" \
    --tag "${LOCAL_IMAGE}" \
    --label "git-commit=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    --label "build-date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --label "plan=hermes-001-E" \
    "${REPO_ROOT}"

echo "    OK — build complete."
echo ""

# ---------------------------------------------------------------------------
# Step 4: Tag for ECR
# ---------------------------------------------------------------------------
echo "[4/5] Tagging: ${LOCAL_IMAGE} → ${FULL_URI}"
docker tag "${LOCAL_IMAGE}" "${FULL_URI}"
echo "    OK."
echo ""

# ---------------------------------------------------------------------------
# Step 5: Push
# ---------------------------------------------------------------------------
echo "[5/5] Pushing: ${FULL_URI}"
docker push "${FULL_URI}"
echo "    OK — push complete."
echo ""

echo "================================================"
echo "Image pushed successfully:"
echo "  ${FULL_URI}"
echo ""
echo "Next step — run terraform plan (PAUSE before apply):"
echo "  cd ${REPO_ROOT}/infra/terraform/hermes-fargate"
echo "  terraform init"
echo "  AWS_PROFILE=${AWS_PROFILE} terraform plan -out=plan-001-E.tfplan"
echo "================================================"
