import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
f = {"name": "get_weather", "description": "Get weather", "parameters": {"type": "OBJECT", "properties": {"loc": {"type": "STRING"}}}}
config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**f)])],
    temperature=0.1
)
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents="Get the weather for London",
    config=config
)
print("Response text:", response.text)
for p in response.candidates[0].content.parts:
    print(p)
