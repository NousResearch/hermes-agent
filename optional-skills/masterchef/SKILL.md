---
name: masterchef
description: "Recipe and ingredient query assistant. Use when users ask how to cook X, X recipe, what can I make with X, I have X what can I cook, how to make X, or mention cooking, recipe, ingredients, dish, food preparation, meal, cuisine. Supports searching by ingredients or by dish name. Also triggers on Chinese keywords: 怎么做X, 做法, 食材, 菜谱, 做菜, 烹饪, 料理, 食谱."
version: 1.0.0
allowed-tools: [Bash]
---

# MasterChef - Recipe Assistant

MasterChef is a cooking assistant powered by the api.yummy.chat recipe knowledge base.

## Overview

MasterChef provides two core features:

1. **Ingredient Search**: Tell me what ingredients you have (1-3 items), and I'll show you what dishes you can make
2. **Recipe Lookup**: Tell me a dish name, and I'll give you the full recipe

## When to Use This Skill

Activate this skill when the user's request involves:

- Asking about recipes: "how to make mapo tofu", "recipe for cola chicken wings", "how to cook fried rice"
- Asking about ingredient pairings: "what can I make with eggplant and potato", "I have chicken wings, what can I cook"
- Expressing cooking intent: "want to eat braised pork", "what should I cook today"
- Mentioning cooking keywords: recipe, cooking, ingredients, dish, meal, cuisine, food prep

## Language Handling (IMPORTANT)

**The API only accepts Chinese input and returns Chinese output.** You must handle translation:

1. **Input translation**: If the user writes in English, translate ingredient names or dish names to Chinese before calling the API
   - "eggplant" → "茄子", "potato" → "土豆", "chicken wings" → "鸡翅"
   - "cola chicken wings" → "可乐鸡翅", "mapo tofu" → "麻婆豆腐"
   - "how to make stir-fried eggplant" → dish_name: "烧茄子"

2. **Output translation**: Translate the full API response back to the user's language (English)
   - Translate dish names, ingredient lists, cooking steps, tips, and all other content
   - Preserve the original structure and formatting
   - Do NOT omit or add any information during translation

**If the user writes in Chinese**, no translation is needed — pass Chinese directly to the API and return Chinese output as-is.

## API Endpoints

### 1. Ingredient Search API

**Endpoint**: `POST https://api.yummy.chat/ingredients`

**Purpose**: Query dishes that can be made with 1-3 ingredients

**Request format**:
```json
{
  "ingredients": ["食材1", "食材2", "食材3"]
}
```

**Response format**:
```json
{
  "results": [
    {
      "ingredient": "茄子",
      "dishes": ["地三鲜", "油焖茄子", "茄子煎饼", ...]
    }
  ]
}
```

### 2. Recipe Lookup API

**Endpoint**: `POST https://api.yummy.chat/howtocook`

**Purpose**: Query detailed recipe by dish name (ingredients, steps, tips)

**Request format**:
```json
{
  "dish_name": "菜名"
}
```

**Response format**:
```json
{
  "dish_name": "地三鲜",
  "recipes": [
    "地三鲜\n所属菜系：东北菜\n烹饪难度：⭐⭐⭐\n\n食材与调味品\n..."
  ]
}
```

## Usage Guide

### Detecting User Intent

**Ingredient search mode** — when the user:
- Mentions "I have", "I've got", "in my fridge"
- Asks "what can I make with X", "what to cook with X"
- Lists multiple ingredient names

**Recipe lookup mode** — when the user:
- Asks "how to make X", "recipe for X", "how to cook X"
- Says "want to eat X", "make some X"
- Mentions a specific dish name

**Priority**: If unsure, try recipe lookup first (it's more specific).

### 1. Ingredient Search Mode

**Steps**:

1. **Extract ingredient names**: Identify 1-3 ingredients from user input
   - Example: "I have eggplant and potato" → `["茄子", "土豆"]`
   - Example: "what can I make with chicken wings" → `["鸡翅"]`

2. **Call API**:
```bash
curl -s -X POST https://api.yummy.chat/ingredients \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["食材1", "食材2"]}'
```

3. **Parse response**: Extract `dishes` list from each ingredient in `results`

4. **Translate and format output**: See "Output Format Guide" below

### 2. Recipe Lookup Mode

**Steps**:

1. **Extract dish name**: Identify the dish name from user input
   - Example: "how to make Di San Xian" → `"地三鲜"`
   - Example: "want to eat cola chicken wings" → `"可乐鸡翅"`

2. **Call API**:
```bash
curl -s -X POST https://api.yummy.chat/howtocook \
  -H "Content-Type: application/json" \
  -d '{"dish_name": "菜名"}'
```

3. **Parse response**: Extract recipe content from `recipes` array

4. **Translate and format output**: See "Output Format Guide" below

## Output Format Guide

### Ingredient Search Results

**Principles**:
- Group results by ingredient
- Highlight dishes that use multiple queried ingredients together
- Use emoji for readability
- Categorize (recommended, quick & easy, advanced, etc.)
- Translate all dish names and descriptions to the user's language

### Recipe Lookup Results

**Principles**:
- **Show the most common/standard version first**: If the API returns multiple versions, only display the first one (typically the most standard)
- **Return the full recipe as-is**: Do not omit or add any information — translate the complete API response faithfully
- If multiple variants exist, briefly mention them without expanding

## Error Handling

### HTTP 404 - Recipe Not Found

The dish name doesn't exist in the knowledge base. Tell the user the recipe wasn't found, suggest checking the spelling or trying an alternative name.

### HTTP 422 - Parameter Error

Invalid input (wrong number of ingredients, empty string). Guide the user to provide 1-3 ingredient names or a specific dish name.

### HTTP 502 - Service Error

The backend knowledge base service is unavailable. Ask the user to try again later.

### Network Error

Cannot connect to api.yummy.chat. Suggest checking the network connection.

## Examples

### Example 1: Ingredient Search (English)

**User**: "I have eggplant and potato, what can I make?"

**MasterChef behavior**:
1. Detect ingredient search mode
2. Translate ingredients: `["茄子", "土豆"]`
3. Call `/ingredients` API
4. Translate dish names to English
5. Highlight combined dishes like Di San Xian (地三鲜)

### Example 2: Recipe Lookup (English)

**User**: "How to make Di San Xian?"

**MasterChef behavior**:
1. Detect recipe lookup mode
2. Translate dish name: `"地三鲜"`
3. Call `/howtocook` API
4. Translate the full recipe to English (ingredients, steps, tips, health advice)

### Example 3: Chinese Input (No Translation Needed)

**User**: "地三鲜怎么做"

**MasterChef behavior**:
1. Detect recipe lookup mode
2. Call API with `"地三鲜"` directly
3. Return Chinese output as-is

## Technical Details

### API
- Base URL: `https://api.yummy.chat`
- Protocol: HTTPS
- Response format: JSON
- **Chinese-only API**: All input/output is in Chinese

### Tools
- Uses `Bash` tool to execute `curl` commands
- Pre-authorized in `allowed-tools` to reduce permission prompts

### Translation
- Agent handles all translation (input → Chinese, output → user's language)
- No external translation API needed — the agent translates using its own language capabilities
- Preserve emoji, formatting, and structure during translation
