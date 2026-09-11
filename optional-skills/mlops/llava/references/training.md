# LLaVA Training Guide

Guide to training and fine-tuning LLaVA models.

## Training stages

### Stage 1: Feature alignment (Pretraining)

**Purpose**: Align vision encoder with language model

**Data**: 558K image-caption pairs (CC3M subset)

```bash
# Download pretrained projector or train from scratch
bash scripts/v1_5/pretrain.sh
```

**Configuration:**
- Base model: Vicuna-7B or LLaMA-2-7B
- Vision encoder: CLIP ViT-L/14
- Training time: ~20 hours on 8× A100

### Stage 2: Visual instruction tuning

**Purpose**: Teach model to follow visual instructions

**Data**: 150K GPT-generated multimodal instruction data

```bash
# Fine-tune with instruction data
bash scripts/v1_5/finetune.sh
```

**Configuration:**
- Epochs: 1
- Batch size: 128 (across 8 GPUs)
- Learning rate: 2e-5
- Training time: ~24 hours on 8× A100

## Data format

### Instruction data format

```json
[
    {
        "id": "001",
        "image": "path/to/image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat is in this image?"
            },
            {
                "from": "gpt",
                "value": "The image shows a dog playing in a park."
            },
            {
                "from": "human",
                "value": "What breed is the dog?"
            },
            {
                "from": "gpt",
                "value": "It appears to be a Golden Retriever."
            }
        ]
    }
]
```

## Fine-tuning on custom data

### Prepare your data

```python
import json

# Create instruction data
data = []
for image_path, qa_pairs in your_dataset:
    conversations = []
    for q, a in qa_pairs:
        conversations.append({"from": "human", "value": f"<image>\n{q}"})
        conversations.append({"from": "gpt", "value": a})

    data.append({
        "id": str(len(data)),
        "image": image_path,
        "conversations": conversations
    })

# Save
with open("custom_data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Fine-tune script

```bash
#!/bin/bash

# Set paths
DATA_PATH="custom_data.json"
IMAGE_FOLDER="path/to/images"
MODEL_PATH="liuhaotian/llava-v1.5-7b"
OUTPUT_DIR="./checkpoints/llava-custom"

# Fine-tune
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

## LoRA fine-tuning (memory efficient)

```python
from peft import LoraConfig, get_peft_model

# LoRA config
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)

# Train with much lower memory
```

## Hardware requirements

### Full fine-tuning

- **7B model**: 8× A100 (40GB)
- **13B model**: 8× A100 (80GB)
- **Training time**: 20-48 hours

### LoRA fine-tuning

- **7B model**: 1× A100 (40GB)
- **13B model**: 2× A100 (40GB)
- **Training time**: 10-24 hours

## Best practices

1. **Start with pretrained** - Don't train from scratch
2. **Use LoRA for efficiency** - 10× less memory
3. **Quality over quantity** - 1K high-quality > 10K low-quality
4. **Multi-turn conversations** - More engaging than single Q&A
5. **Diverse images** - Cover different scenarios
6. **Clear instructions** - Specific questions get better answers
7. **Monitor loss** - Should decrease smoothly
8. **Save checkpoints** - Training can fail
9. **Test regularly** - Validate on held-out set
10. **Use DeepSpeed** - For multi-GPU training

## Resources

- **Training script**: https://github.com/haotian-liu/LLaVA/tree/main/scripts
- **Data format**: https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md
- **Paper**: https://arxiv.org/abs/2304.08485
