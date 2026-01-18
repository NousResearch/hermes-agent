#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/imagen_eval_gpt5_${TIMESTAMP}.log"

echo "📝 Logging output to: $LOG_FILE"

python batch_runner.py \
  --dataset_file="source-data/hermes-agent-imagen-data/hermes_agent_imagen_train_sft.jsonl" \
  --batch_size=20 \
  --run_name="imagen_train_sft_glm4.7" \
  --distribution="image_gen" \
  --model="z-ai/glm-4.7" \
  --base_url="https://openrouter.ai/api/v1" \
  --providers_allowed="gmicloud,siliconflow,atlas-cloud,z-ai,novita" \
  --num_workers=50 \
  --max_turns=25 \
  --ephemeral_system_prompt="When generating an image for the user view the image by using the vision_analyze tool to ensure it is what the user wanted. If it isn't feel free to retry a few times. If none are perfect, choose the best option that is the closest match, and explain its imperfections. If the image generation tool fails, try again a few times. If the vision analyze tool fails, provide the image to the user and explain it is your best effort attempt." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"
#  --verbose \