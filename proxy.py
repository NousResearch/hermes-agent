import os
import litellm
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure litellm to use the GEMINI_API_KEY from the environment.
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY environment variable must be set before starting the proxy.")

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    try:
        data = request.get_json()
        model = data.get("model", "gemini/gemini-1.5-pro")
        messages = data.get("messages", [])
        
        # Call litellm which handles the translation
        response = litellm.completion(
            model=model if model.startswith("gemini/") else f"gemini/{model}",
            messages=messages,
            stream=False # Keep it simple
        )
        return jsonify(response.model_dump())
    except Exception as e:
        return jsonify({"error": {"message": str(e), "code": 500}}), 500

if __name__ == "__main__":
    app.run(port=4000)
