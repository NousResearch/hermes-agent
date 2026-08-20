import os
from hermes_agent.skills.suprawall import suprawall_check

def main():
      # Example usage of suprawall_check
      tool_name = "browser_open_url"
      arguments = {"url": "https://example.com"}

    result = suprawall_check(tool_name, arguments)

    if result["status"] == "ALLOW":
              print(f"Tool call to {tool_name} is allowed.")
elif result["status"] == "DENY":
          print(f"Tool call to {tool_name} is denied: {result['reason']}")
elif result["status"] == "REQUIRE_APPROVAL":
          print(f"Tool call to {tool_name} requires approval.")

if __name__ == "__main__":
      main()
  
