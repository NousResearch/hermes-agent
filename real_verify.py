
import asyncio
from run_agent import AIAgent

async def real_test():
    # This callback mirrors the TUI Gateway's behavior
    def usage_callback(usage):
        print(f"LIVE_UPDATE: {usage}")

    # Initialize agent with the real provider configuration
    agent = AIAgent(
        model="gemma-4-31b-it", 
        usage_callback=usage_callback
    )

    print("Sending prompt to trigger streaming...")
    # We use a prompt that forces a moderately long response to see the counter move
    try:
        # run_conversation usually handles the loop and streaming internally.
        # To see the live updates, we need to make sure the agent is actually streaming.
        # The AIAgent.run_conversation method by default uses the internal loop.
        
        # Let's use a simple prompt.
        response = agent.run_conversation("Write a 3-paragraph essay on why token estimation is important for AI TUIs.")
        print(f"\nFinal Response received. Length: {len(response)}")
    except Exception as e:
        print(f"Error during conversation: {e}")

if __name__ == "__main__":
    asyncio.run(real_test())
