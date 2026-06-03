// Supabase Edge Function (synthetic tiny shape, not copied real code)
// Deno/Supabase runtime shape only
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';

serve(async (req) => {
  const { name } = await req.json();
  const data = { message: `Hello, ${name}! (fixture)` };
  return new Response(JSON.stringify(data), {
    headers: { 'Content-Type': 'application/json' },
  });
});
