import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

serve(async (req) => {
  // create-invite-code edge function
  return new Response(JSON.stringify({ code: "abc123" }), {
    headers: { "Content-Type": "application/json" },
  });
});
