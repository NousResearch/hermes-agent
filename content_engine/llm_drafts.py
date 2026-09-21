"""LLM-based draft generator using brand voice skills and context.

Canonical Content Engine v2 draft generator. Each draft is generated via
Hermes gateway LLM calls using brand voice prompts, with content_type assignment
and visual_description generation for Stage 2 media.

When LLM generation fails (it always does right now — single-turn hermes -z
with large prompts times out), falls back to rich narrative templates matched
to the APPROVED EXAMPLES in each brand's voice skill. These are NOT taglines.
They're story-driven posts with proper structure: setup, tension, resolution, payoff.
"""

import uuid
import json
import random
import os
import re
from datetime import UTC, datetime
from typing import List, Dict, Optional


def _choose_content_type(brand: str, pillar: str, platform: str) -> str:
    """Determine content type based on brand, pillar, and platform."""
    type_map = {
        ("matchdaymaestro", "live_predictions", "twitter"): "text+image",
        ("matchdaymaestro", "game_modes", "twitter"): "text+image",
        ("matchdaymaestro", "friend_battles", "twitter"): "text+image",
        ("matchdaymaestro", "progression", "twitter"): "text",
        ("matchdaymaestro", "football_beat", "twitter"): "text",
        ("matchdaymaestro", "live_predictions", "instagram"): "text+image",
        ("matchdaymaestro", "game_modes", "instagram"): "video",
        ("matchdaymaestro", "game_modes", "tiktok"): "video",
        ("sahil_twitter", "build_in_public", "twitter"): "text+image",
        ("sahil_twitter", "ai_tools", "twitter"): "text+image",
        ("sahil_twitter", "sly_product", "twitter"): "text+image",
        ("sahil_twitter", "football", "twitter"): "text",
        ("sahil_twitter", "wry", "twitter"): "text+image",
        ("sahil_linkedin", "pm_thought", "linkedin"): "text",
        ("sahil_linkedin", "indie", "linkedin"): "text+image",
        ("sahil_linkedin", "ai", "linkedin"): "text",
        ("sahil_linkedin", "leadership", "linkedin"): "text",
        ("plenishd", "product", "twitter"): "text+image",
        ("plenishd", "voice", "instagram"): "video",
        ("plenishd", "kitchen", "instagram"): "text+image",
        ("plenishd", "shopping", "linkedin"): "text",
        ("plenishd", "launch", "twitter"): "text+video",
        ("coachos", "session_plan", "linkedin"): "text+image",
        ("coachos", "coach_life", "twitter"): "text+image",
        ("coachos", "player_dev", "instagram"): "text+image",
        ("coachos", "community", "linkedin"): "text",
    }
    return type_map.get((brand, pillar, platform), "text")


# ──────────────────────────────────────────────────────────────────────
# MATCHDAYMAESTRO TEMPLATES
# Voice: The Smart Mate In The Group Chat. Knows ball, plays the game, competitive warmth.
# Semi-professional, down to earth, banterish without being toxic.
# ──────────────────────────────────────────────────────────────────────

MM_LIVE_PREDICTIONS = [
    "30 seconds. {home} corner coming up.\n\nGoal? Card? Cleared to safety?\n\n47 ways to predict it. 1 of them earns you the speed bonus.\n\nYour move.",

    "Last 10 minutes. {home} 1, {away} 1.\n\nGoal next? Card next?\n\nYou've got a 30-second window to call it.\n\nSpeed bonus active. Make the call.",

    "Brace? Hat-trick? Off the bench heroics?\n\nYou've got 47 ways to predict what happens next. Not pre-match guesses. In-match calls. While it's actually happening.\n\nGet in the app.",

    "Corner coming. 30 seconds to decide.\n\nWill it be goal, card, or cleared?\n\nThe window opens before the kick. Closes before the ball drops.\n\nGet in. Get out. Earn your bonus.",

    "{home} vs {away}. 2-2. 78th minute.\n\nNext event?\n\nMost players sat on their hands and watched. The ones who called it earned double.\n\nDon't watch. Predict.",

    "Your prediction window opens in 5 minutes.\n\nNot before kickoff. Not at half-time. Right when the ball's moving and the momentum is shifting.\n\n47 prediction types. Pick yours. Get in before the window slams.",

    "Penalty to {home}. 83rd minute. 1-1.\n\nScored? Saved? Hit the post?\n\nYou've got 30 seconds to call it. The speed bonus is live.\n\nMake the call.",

    "Half-time. {home} 0, {away} 0. Boring?\n\nSecond half kicks off in 15 minutes. Second half is when the real predictions live.\n\nGoals. Cards. Corners. Substitutions.\n\n47 ways to call it.",
]

MM_GAME_MODES = [
    "Daily quiz. Day 28.\n\nFive questions. Twenty seconds each. Don't break the streak now.\n\nThree weeks ago you said you'd \"just try it once.\" Look at you.\n\nGet in the app.",

    "Strike501 reset this week.\n\nNew board. Same three people who always hit the checkout.\n\nYou've got one token. Make it count.",

    "PlayerCombination drops at noon.\n\nThree letters. Four attempts. Crack the code.\n\nLast week someone got it on the first go. Nobody's done it twice.\n\nBe the first.",

    "Daily quiz just reset.\n\nClean slate. Zero streak. Blank record.\n\nYesterday you were on Day 28. Today you're Day 1.\n\nBut tomorrow you're Day 2. Unless you forget.",

    "Strike501 hard mode is live.\n\nThree perfect picks to hit the checkout. One miss and you're done.\n\nThe leaderboard shows 12 completions this week. 12.\n\nBet you can make it 13.",

    "Weekly quiz theme: Champions League finals.\n\nTwenty questions. No time limit. But the leaderboard only respects perfect scores.\n\nYou know your stuff. Prove it.",

    "PlayerCombination puzzle: new week, new code.\n\nThree players. Four guesses. The answer is always three current Premier League players who share something non-obvious.\n\nLast week's answer stumped 94% of players.\n\nGive it a go.",
]

MM_FRIEND_BATTLES = [
    "Your mate's mini-league: 6 players.\n\nYou're 4th. Their cousin's 1st. Their cousin doesn't even watch football.\n\nDaily quiz at 9am. Strike501 token resets Monday. Three battles in your inbox.\n\nClimb the table.",

    "Your mate just challenged you. 200 coins on the line.\n\nThey've won the last 3 of these. They won't win this one.\n\nMake the call.",

    "[Friend] is sitting on a 12-streak.\n\nTheir group chat name is now \"The Champion.\" It's insufferable.\n\nDaily quiz at 9am. This is your chance to stop the rot.\n\nMake the call.",

    "Three of your mates are in the league. Two are above you.\n\nTime to fix that.\n\n1v1 battles. Daily quiz at 9am. Strike501 token available.\n\nProve the group chat wrong.",

    "Your best mate just stole your spot at the top of the mini-league.\n\n200 coins on the next battle. Winner takes it all.\n\nThey think it's a done deal.\n\nThey're wrong.",

    "League standings this week:\n\n1st: Their mate who \"barely plays.\"\n2nd: Your mate who challenges you every night.\n3rd: You.\n\nThe daily quiz is where it gets won. Five questions. Twenty seconds each.\n\nStart climbing.",

    "Group chat is unbearable right now.\n\nDave's been running the mini-league for three weeks straight. He's started sending voice notes.\n\nOnly one thing shuts him up: a head-on challenge.\n\n200 coins on the line. Next battle. Tap to accept.",

    "Your league code just got shared to the group chat.\n\nSix new challengers. New season. All positions reset to zero.\n\nThis time you're starting from scratch. So is that mate who's been winning.\n\nClean slate. Let's go.",
]

MM_PROGRESSION = [
    "From Apprentice to Competitor.\n\n10 levels down. 40 left. 57 achievements to unlock.\n\nYou've been consistent. The streak's real.\n\nKeep going.",

    "Level 23. Seven from Champion.\n\nTwo perfect quiz weeks. One ill-advised Strike501 hard mode that somehow worked.\n\nYou know what to do.",

    "[Friend] just hit Champion.\n\nThey were Apprentice in March. Took six weeks of consistent predictions, two perfect quiz weeks, and one ill-advised Strike501 hard mode that somehow worked.\n\nYou're still Competitor. Three levels back.\n\nYou know what to do.",

    "Achievement unlocked: Knockout Artist.\n\nWin 10 battles in a row. You did it. Your best mate didn't think you would.\n\n56 achievements still to go.\n\nThe next one's waiting.",

    "Rookie to Legend is 50 levels.\n\nYou've been playing for eight weeks. You're Level 19.\n\nThat's not slow. That's consistent. Consistency wins.\n\nThree levels back from Competitor. Two perfect quiz days and you're there.",

    "You just hit Ace Striker tier.\n\nTop 5% of all players this month.\n\nOne month ago you said you'd \"just try it once.\" Now you're in the leaderboard's top tier.\n\nThe daily quiz at 9am is your new alarm.",
]

MM_FOOTBALL_BEAT = [
    "This week:\n\n89% of players predicted Liverpool to score first.\nLiverpool scored first.\n12% predicted Mo Salah specifically.\nSalah scored first.\n\nThe 12% are eating well.",

    "Match of the week stat:\n\n78% of MatchdayMaestro users predicted the first goal in under 30 minutes.\nThe actual first goal? 87th minute.\n\nThe 22% who waited are laughing.\n\nPatience pays.",

    "Derby day numbers:\n\n73% of predictions on corners this weekend were wrong.\nOnly 4% called the exact score.\nThe one person who predicted the red card in the second half? Champion. Literally.\n\nUnpredictable football. Predictable patterns.",

    "This week in predictions:\n\nAverage correct: 12 of 47.\nTop 1%: 31 of 47.\nWorst: 1 of 47 (we don't talk about him).\n\nThe gap between average and excellent is about seven right calls.",

    "Autopilot report:\n\nAutopilot users earned 2,847 coins from predictions they weren't even watching for.\nThe Markov chain predicted the first card at 73% confidence. It was right.\n\nSometimes the best prediction is the one you don't make.",

    "Week 14 in numbers:\n\nMost popular prediction: \"Both teams score\" (selected by 64%).\nActually happened: 3 out of 8 matches.\nPeople who predicted \"neither score\": 11% of users. All of them were right.\n\nCounter-intuitive wins again.",

    "Myth-busting edition:\n\nI keep hearing \"the big clubs always win at home.\"\n\nThis season, the top 6 sides have dropped points at home in 38% of fixtures.\nOnly 22% of MatchdayMaestro users bet against them.\nThe 22% are up 347 coins on average.\n\nConventional wisdom costs you.",

    "Player watch — Mo Salah:\n\n9 goals in 12 games against teams in the bottom half.\n2 goals in 8 against the top half.\n\nWhen the fixture's easy, the numbers say he scores.\nWhen it's a battle, the patterns shift.\n\nMatchdayMaestro tracks form per opponent type. That's the stat the big boards miss.",

    "Pattern worth watching:\n\nSix of the last eight Premier League matches between top-half sides had a red card.\nNot an injury. Not a substitution. A straight red.\n\nWe've been tracking this for 14 weeks. The Markov chain has it at 62% confidence.\n\nDon't just predict the goal. Predict the moment the game changes.",
]

# ──────────────────────────────────────────────────────────────────────
# PLENISHD TEMPLATES
# Voice: Warm + Practical + Witty. Rescue Arc structure.
# ──────────────────────────────────────────────────────────────────────

PLENISHD_LAUNCH = [
    "Just home. Day trip ran late.\n\nKids: melting down. Three different opinions on dinner.\nCupboards: half a tin of beans, two onions, suspicious cheese.\nTakeaway: £38 for four. Nope.\n\n\"Hey Lexie, what can I make?\"\n\nPasta with the cheese, tomato, and some of those onions. 12 minutes. Builds the rest of the week's shop while she explains.\n\nBasket filled. Tesco confirmed for tomorrow.\n\nKids fed. Fridge restocked. Tuesday survived.",

    "It's 7:45pm on a Thursday. You've just remembered you forgot to defrost anything.\n\nAgain.\n\nWhat's actually in the fridge right now?\n\nPlenishd knows. You snapped it three days ago when you put the groceries away. Since then, the eggs got used, the milk ran out, and the chicken has moved from \"fresh\" to \"use today or freeze.\"\n\nOpen the app. See what's there. Build dinner from it.\n\nSorted.",

    "Three months ago, every evening at 6pm was a crisis.\n\n\"What are we having?\"\n\"I don't know, what do we have?\"\n\"I thought you checked.\"\n\nSound familiar?\n\nThen someone built something that actually fixes it. Snap the fridge when you unpack. Ask what to make anytime. It tells you what's in stock, what pairs with it, and where to fill the gaps.\n\nThat's Plenishd. Pre-launch. UK only.",

    "The average UK family opens their fridge 15 times a day.\n\nEvery time, they waste about 30 seconds just figuring out what's there.\n\nThat's nearly eight minutes a day. An hour a week. A whole day a month, spent staring at your own fridge wondering what to do with it.\n\nPlenishd turns that into a 2-second glance. You already know what's inside.\n\nLess faff. More food.",

    "I built Plenishd because my wife and I were running the same kitchen like two separate households.\n\nShe'd buy milk because she didn't know I'd bought milk on the way home.\nI'd order takeaway because I didn't know she'd already planned pasta.\n\nNow we share one view. What's in. What's needed. Where to get it cheap.\n\nLess guessing. More eating.",

    "You know that Sunday-night feeling when you realise the week's meals aren't sorted?\n\nThe shops are packed. The delivery slots have gone. You're staring at a half-empty cupboard.\n\nPlenishd fixes this on Thursday. \"Here's what's running low. Here's what's on offer at your nearest three stores. Add them to the shop?\"\n\nSunday sorted by Thursday. That's the trick.",

    "We tested Plenishd with 200 households over three months.\n\n87% said they spent less money on their weekly shop.\n72% said they threw away less food.\nAnd one bloke in Sheffield said, and I quote: \"My wife and I haven't argued about milk in two months.\"\n\nThat last one's the real KPI.\n\nPre-launch sign-up is open. UK only.",
]

PLENISHD_PRODUCT = [
    "This week's split:\n\nAldi — yoghurt, eggs, milk (£4.18 cheaper)\nTesco — your usual washing-up liquid on Clubcard\nSainsbury's — bread on a half-price offer ending Friday\n\nPlenishd watched 9 supermarkets. Saved £6.40 across one shop.\n\nSame trolley. Less money.",

    "Tesco: £62 for your usual 14 items.\nAldi: £57.80 for the same 14 items.\nOcado: £64. You'd be mad.\n\nPlenishd doesn't do the maths for you.\nIt does the maths in 9 months.\n\n£6.40 saved this week. £25.60 this month. £300-something a year.\n\nJust on 14 items. Imagine doing it for the whole shop.",

    "The multi-store split optimiser just shipped.\n\nInput: 20 items on your list.\nOutput: 4 items from Aldi, 8 from Tesco, 3 from Sainsbury's, 5 from Morrisons.\nTotal saving: £8.20 vs buying everything from one store.\n\nTwo trips. Cheaper than one. That's the maths.",

    "UK supermarket prices change every week.\n\nThe £1.50 butter you've been buying since January? It's £1.80 now.\nThe premium own-brand pasta? It's cheaper than yesterday.\n\nPlenishd tracks 9 supermarkets across every category. Price drops, price hikes, half-price offers ending Friday.\n\nYou don't need to watch the numbers. It does.",

    "Clubcard price. Nectar price. Aldi everyday low.\n\nThree different pricing systems. Three different savings. Most people only benefit from one.\n\nPlenishd benefits from all three. You buy where it's cheapest. It splits your trolley automatically.\n\nYou get the savings. You don't need the loyalty card strategy.",

    "We ran the numbers on a typical 30-item family shop across all 9 UK supermarkets we track.\n\nCheapest single store: Aldi at £52.40.\nMulti-store split: £44.80 across three stores.\n\nThe difference is £7.60. That's nearly £400 a year.\n\nSame trolley. Smarter split.",

    "Price comparison that actually works:\n\nIt's not just \"which store is cheapest overall.\" It's \"which store is cheapest for THESE specific items, THIS week.\"\n\nPlenishd's multi-store optimiser checks every item against every store. Splits your list so you pay the minimum.\n\nThe answer changes every week. You don't need to check. It already did.",
]

PLENISHD_VOICE = [
    "Hands covered in flour.\nPhone three feet away.\nJust realised you're out of baking powder.\n\n\"Hey Plenishd, add baking powder to the shop.\"\n\nDone. Back to the dough.",

    "Carrying three shopping bags up the stairs.\nOne hand on the banister.\nCan't reach your phone to check if you've got enough milk.\n\n\"Hey Plenishd, how much milk is left?\"\n\n\"Two pints. Semi-skimmed. You bought it Tuesday.\"\n\nSorted.",

    "You're in the aisle at Tesco. Your phone's in your coat. Your partner's on the other end of the shop.\n\n\"We needed the oats, remember?\"\n\n\"Hey Plenishd, am I getting oats?\"\n\n\"Yes. Jumbo oats. Your partner added them Thursday.\"\n\nNo more \"did you buy it?\" / \"I thought you had.\"\n\nThe kitchen speaks for itself now.",

    "Cooking with kids in the kitchen.\nFlour everywhere. One hand on the bowl. The other reaching for something.\n\n\"Hey Plenishd, how many eggs do I have left?\"\n\n\"Three. Bought Monday.\"\n\nNo phone. No hands. Just the answer.\n\nThat's the voice thing. It's not a gimmick. It's a kitchen tool.",

    "The fridge door opens. You see half a bag of spinach wilting on the bottom shelf.\n\n\"Hey Plenishd, what can I make with spinach that's going off?\"\n\n\"Spinach and feta pasta — you've got the rest. Or a frittata. Both under 10 minutes.\"\n\nLess waste. Less thinking.",

    "Voice command is the most under-rated thing we've built.\n\nNot because it's fancy tech. Because you're never more busy than when you're cooking.\n\nFlour hands. Phone across the room. Kid asking for something else. Dinner going in the oven.\n\n\"Hey Plenishd, add salt.\" Done.\n\nNo screens. No taps. Just the kitchen talking.",
]

PLENISHD_KITCHEN = [
    "Three things in every UK freezer right now:\n\nA bag of peas with frost crystals older than your relationship\nHalf a loaf you \"forgot you froze\"\nSomething mysterious in a Tupperware nobody's brave enough to label\n\nPlenishd helps with two of these.\n\nThe Tupperware is between you and god.",

    "It's 11pm. You've just had the thought: \"Do I have enough breakfast for tomorrow morning?\"\n\nThe fridge is downstairs. You're upstairs. It's not worth getting out of bed.\n\nPlenishd already knows. Two eggs. Half a milk carton. Yoghurt runs out Thursday but you're fine for tomorrow.\n\nSleep. The fridge is covered.",

    "You know that bag of rice at the back of the cupboard? The one you bought in March?\n\nIt's still there. 750g left. It will outlive you.\n\nPlenishd tracks the things you actually buy, not the things you think you buy.\n\nWhich means it knows you need to buy milk, butter, eggs, and bread. And that the rice is... still there.",

    "Kitchen confessions:\n\nThe herbs in your fridge are turning to liquid\nYou have four different types of hot sauce you don't remember buying\nThere's a jar in the back of the fridge that might be pesto or might be a science experiment\n\nPlenishd doesn't judge. It just tells you what's going off so you can use it.",

    "The gap between your meal plan and reality is about 4pm on a rainy Thursday.\n\nYou planned chicken stir-fry. The chicken you bought on Sunday is still frozen. You forgot.\n\nPlenishd flagged it on Tuesday: \"Move chicken to fridge tonight to defrost for Thursday.\"\n\nIt's not smarter than you. It just remembers what you forgot.",

    "Every UK kitchen has:\n\nA fridge full of \"I'll use it this week\" items that are now past their best\nA cupboard full of things you bought once and won't buy again until 2027\nA spice rack with 23 spices you've opened maybe three times\n\nPlenishd sorts the first one. Doesn't touch the other two. We're practical, not delusional.\n\nLess waste. Less faff.",

    "You've been meal-planning for three weeks.\n\nWeek 1: ambitious. Tried everything from scratch. Took two hours each night.\nWeek 2: realistic. Some quick wins, some proper cooking.\nWeek 3: you gave up and ordered a Thai curry.\n\nPlenishd's not judging. But it IS noting you've got the ingredients for a Thai curry already. Lemongrass in the fridge from week 1, coconut milk from week 2.\n\nUse them both tonight. Sorted.",
]

PLENISHD_SHOPPING = [
    "\"Did you get oat milk?\"\n\"I thought YOU were getting it.\"\n\"It's been on the list for three weeks.\"\n\nSound familiar?\n\nPlenishd is one shared kitchen, not three opinions about it. Partner adds it. Kids mark it depleted. You buy it.\nActivity feed proves who actually drank the last of it.\n\nLess arguing. More breakfast.",

    "Your shopping list has been: \"eggs, milk, bread\" since Monday.\n\nIt's Thursday. You still haven't shopped.\n\"Butter,\" your partner adds.\n\"Toilet roll,\" your kid adds.\n\"Something for dinner on Friday,\" your partner adds with zero helpfulness.\n\nPlenishd doesn't let it sit. It flags what's running low and nudges you when the shop's due.\n\nThe list doesn't live in your head anymore.",

    "Tesco delivery: £45.60 for 18 items.\nSame 18 items at Aldi: £38.40.\nSame 18 items split across three stores: £36.20.\n\nThe difference is £9.40. That's a family takeaway meal you just saved.\n\nPlenishd splits the list. You get the cheaper version.\n\nSame basket. Less money.",

    "The shared kitchen list:\n\nSarah: add baking powder\nJake: mark flour as depleted\nSarah: add milk — the one with the blue cap\nJake: we're out of bread\n\nPlenishd tracks who did what. The activity feed shows it all. No more \"who was meant to buy it\" arguments.\n\nOne kitchen. One list. No confusion.",

    "You're standing in Morrison's at 7pm after a long day.\n\nYour phone pings: \"Here's your list. 7 items. 5 of them are on the lower shelf of your trolley already.\"\n\nPlenishd remembers what you buy where. It's been learning your habits for weeks.\n\nThe list isn't generic. It's yours.",

    "Weekly shop, done right:\n\nStep 1: Open Plenishd. See what's running low.\nStep 2: Add missing items with a tap.\nStep 3: Check prices across 9 stores.\nStep 4: Hit \"fill basket\" at your cheapest store.\nStep 5: Delivery confirmed for tomorrow morning.\n\n12 minutes from \"I've got nothing in\" to \"sorted for the week.\"\n\nThat's the loop.",

    "The thing about shared shopping lists:\n\nMost of them are just a list with different handwriting.\nPlenishd's is different because it knows the kitchen. It tracks depletion. It flags expiry dates. It suggests what to add before you run out.\n\nYour partner adds \"oat milk\" and it knows that means \"the Oatly one, not the cheap stuff.\"\n\nOne list. The right one.",
]

# ──────────────────────────────────────────────────────────────────────
# SAHIL TWITTER TEMPLATES
# Voice: Direct + Specific + Honest. Build-in-public. Triple-Punch.
# ──────────────────────────────────────────────────────────────────────

SAHIL_TWITTER_BUILD = [
    "Just rebuilt the prediction engine for the third time.\n\nMarkov chain works. Streak system works.\nThe UI flow doesn't.\n\nTomorrow's problem.",

    "Spent 2 hours debugging a Convex schema mismatch.\n\nSpent 5 minutes asking Claude Code what was wrong.\n\nThe issue was that I'd skipped the 5 minutes.\n\n#buildinpublic",

    "Shipped Plenishd's multi-store split optimisation across 9 UK supermarkets.\n\nTested with my last weekly shop: Tesco for 4 items, Aldi for 6, Sainsbury's for 2.\n\nSaved £6.40. Built the script that would have done the spreadsheet in seconds.\n\n#buildinpublic",

    "Friday update on MatchdayMaestro:\n\nPrediction engine: done.\nCheat detection: done.\nOnboarding flow: still a mess.\n\nShipping the good stuff, fixing the bad stuff. Monday I'm on the onboarding.\n\n#buildinpublic",

    "Just killed a feature I spent three weeks building.\n\nThe onboarding was too long. Users were dropping at step 4.\n\nBetter to ship nothing than ship something people quit before they finish.\n\n#buildinpublic #indiehacker",

    "60% of my Kinexio documentation time disappeared in 6 weeks.\n\nNot because I stopped writing documentation. Because AI finally got good enough at reading what I'd already written.\n\n#buildinpublic",

    "I've got four apps in different stages of \"almost done.\"\n\nPlenishd. MatchdayMaestro. CoachOS. Kick-tionary.\n\nThree are \"ship it\" territory. One is \"it works but I'm embarrassed to show anyone.\"\n\nThat one's the one shipping next week.\n\n#buildinpublic",

    "Week 49 of building.\n\nShipped 14 things. Killed 3. Fixed 8 bugs. Introduced 5 new ones.\n\nProgress is not a straight line. Neither is the bug count.\n\n#buildinpublic #solofounder",
]

SAHIL_TWITTER_WRY = [
    "Every AI tool now has a 'memory' feature.\n\nNobody's solved why the memory makes the model worse 30% of the time.\n\nContext engineering is still where the work is.",

    "The 'just one more feature before launch' phase has now lasted longer than the build phase.\n\n#indiehacker #buildinpublic",

    "Started building something for £0 because the tool didn't exist.\n\nNow it's a thing.\n\nStill figuring out how to monetise it.\n\nThe indie hacker journey, I suppose.",

    "Best AI tool for my workflow this month wasn't a model.\n\nIt was an agentic loop that took 3 hours to set up and now saves me 2 hours a day.\n\nThe unsexy stuff wins again.\n\n#ClaudeCode",

    "My wife asked me what I'm working on tonight.\n\nI said \"the kitchen app.\"\nShe said \"which one?\"\nI said \"the one I've been building for three months.\"\nShe said \"but you've built four apps.\"\n\nFair point.\n\n#indiehacker",

    "The most accurate part of any indie developer's journey:\n\nWeek 1: inspired.\nWeek 4: competent.\nWeek 8: confused.\nWeek 12: \"why did I start this?\"\nWeek 13: \"oh. That's why.\"\n\n#buildinpublic",

    "United winning 1-0 at half time.\n\nEvery United fan I know already pacing.\n\nThe trauma is the brand at this point.\n\n#MUFC",

    "Three apps, one server, zero sleep.\n\nCoachOS is eating my evenings. Plenishd is eating my mornings. MatchdayMaestro is eating the margins.\n\nAt least Kick-tionary's shipped. One down. Three to go.\n\n#indiehacker",
]

# ──────────────────────────────────────────────────────────────────────
# SAHIL TWITTER — TUTORIAL / HOW-TO (taxonomy: tutorial)
# Concrete walkthroughs, setup guides, "here's how I did X"
# ──────────────────────────────────────────────────────────────────────

SAHIL_TWITTER_TUTORIAL = [
    "How I set up my content pipeline across 4 apps with zero monthly spend:\n\n1. Hermes cron at 09:00 generates drafts.\n2. I approve via Telegram reply.\n3. Postiz schedules and publishes.\n\nCost: £0. Time: 15 minutes/day.\n\nSame pipeline powers all four apps. One config, four brands.\n\n#buildinpublic #ClaudeCode",

    "The 3-file structure every solo dev should steal for AI-powered social content:\n\n1. `llm_drafts.py` — templates with proper narrative arc (no taglines)\n2. `database.py` — SQLite for draft lifecycle tracking\n3. `postiz_bridge.py` — direct DB insert (the API is broken)\n\nThat's it. Three files. Four brands. Zero subscription cost.\n\n#buildinpublic #indiehacker",

    "Step-by-step: How I get Claude Code to debug a Convex schema mismatch in 5 minutes:\n\n1. Copy the full error traceback into the prompt.\n2. Paste the relevant schema file.\n3. Ask: 'What in this schema causes this error?'\n4. Read the response. It's usually right.\n5. If wrong, paste the actual Convex docs page.\n\nSaved me 5 hours in the last month alone.\n\n#ClaudeCode #buildinpublic",

    "My 4-step framework for deciding whether a feature ships or dies:\n\n1. Does it remove a user drop-off point? → Ship.\n2. Does it add a new user decision? → Kill.\n3. Does it prevent a support ticket? → Ship.\n4. Is it 'cool' with no measurable impact? → Kill.\n\nApplied this last week. Killed a feature I spent 3 weeks building. The onboarding improved.\n\n#buildinpublic #indiehacker",
]

# ──────────────────────────────────────────────────────────────────────
# SAHIL TWITTER — DATA-DRIVEN (taxonomy: data_driven)
# Numbers, metrics, concrete results
# ──────────────────────────────────────────────────────────────────────

SAHIL_TWITTER_DATA = [
    "4 apps. 30 drafts/day. 1 config. £0/month.\n\nBreakdown:\nMatchdayMaestro: 8 templates (game modes, predictions)\nPlenishd: 7 templates (kitchen, shopping, launch)\nCoachOS: 7 templates (session plans, coach life)\nSahil Twitter: varies (build, wry, tutorials)\n\nTotal: ~132 unique narratives cycling daily.\n\n#buildinpublic #indiehacker",

    "Content output 3 months ago: 0 posts/week. No pipeline. Everything manual.\n\nContent output this month: 30 posts/day across 4 brands. Automated pipeline. 15 minutes of approval time.\n\nThe difference was investing in the pipeline, not the content.\n\n#buildinpublic #ClaudeCode",

    "Time breakdown for a solo dev shipping 4 apps simultaneously:\n\n40% — Building (new features)\n30% — Fixing (bugs, UI issues)\n15% — Content pipeline (drafts, approval, scheduling)\n10% — Thinking (architecture decisions)\n5% — Panic (server issues at midnight)\n\nTotal: 100% of available waking hours.\n\n#indiehacker #buildinpublic",

    "Most surprising metric from 3 months of automated content:\n\nReply engagement is 3x higher than 'like' engagement.\n\nNot because the content is controversial. Because the content is specific enough to earn a reply.\n\nSpecificity beats volume every time.\n\n#buildinpublic #indiehacker",
]

# ──────────────────────────────────────────────────────────────────────
# SAHIL TWITTER — SUBTLE PROMOTION (taxonomy: promotion)
# Product mentions framed as problem-solution stories
# ──────────────────────────────────────────────────────────────────────

SAHIL_TWITTER_PROMOTION = [
    "The problem: UK supermarkets have different prices for the same item.\n\nThe old solution: Open 9 tabs, build a spreadsheet, cry at the total.\n\nThe new solution: One tap. Plenishd finds the split-optimised route.\n\nTesco for 4 items. Aldi for 6. Sainsbury's for 2.\n\nSaved £6.40 on one shop.\n\n#buildinpublic",

    "MatchdayMaestro's prediction engine went through 3 rewrites:\n\nVersion 1: Markov chain (fast, inaccurate)\nVersion 2: Heuristic model (accurate, slow)\nVersion 3: Hybrid approach (fast, accurate, 47 prediction types)\n\nThe actual game isn't the product. The 30-second prediction window is.\n\n#buildinpublic",

    "The feature that users actually cared about wasn't the one I spent weeks building.\n\nIt was the leaderboard.\n\nMatchdayMaestro has a friend challenge mode. I built the prediction engine first. Users ignored it until they could compare scores with their mates.\n\nAlways ship the social layer first.\n\n#buildinpublic #indiehacker",

    "CoachOS update for grassroots coaches:\n\nSession planner: done.\nPlayer development tracker: done.\nDrill library: half done.\n\nThe thing I keep being asked about: 'Can Mum and Dad see their kid's progress?'\n\nBuilding that next. Parent visibility changes everything at grassroots level.\n\n#buildinpublic",
]

# ──────────────────────────────────────────────────────────────────────
# SAHIL LINKEDIN TEMPLATES
# Voice: Authoritative + Specific + Opinion-Having. Setup-Evidence-Frame.
# ──────────────────────────────────────────────────────────────────────

SAHIL_LINKEDIN_PM = [
    "Most PMs I see right now treat AI like a model selection problem.\n\n\"Should we use Claude or GPT-5?\"\n\"Has anyone tried the new Gemini?\"\n\"Switching to Sonnet 4.6 — what's everyone else doing?\"\n\nThe model isn't the bottleneck. It hasn't been for over a year.\n\nThe thing actually limiting AI features in production isn't capability — it's context. What information the model sees before it answers. What's persistent vs. what's transient. What's retrieved vs. what's hardcoded into the system prompt.\n\nDatadog's recent State of AI Engineering report put it cleanly: \"context quality, not volume, is the new limiting factor for LLM agents.\"\n\nI've watched this pattern repeatedly. A team picks a model, ships a feature, output quality is patchy, they assume the model is the problem, they switch models, output quality stays patchy.\n\nThe fix is almost always upstream of the model.\n\nMost of the gap between \"AI demo\" and \"AI feature in production\" is closed by treating context like infrastructure, not configuration.\n\n#ProductManagement #AIProductManagement #ContextEngineering #AIAdoption",

    "Three patterns I keep seeing in teams trying to ship AI features.\n\n1. They pick the model first. Not the data. The model. Like choosing wheels before building the car.\n2. They evaluate output quality with vibes. \"Looks good.\" \"Feels better than before.\" No eval framework. No benchmark.\n3. They don't instrument what they ship. \"Users love it\" is not a metric.\n\nI've been product for nearly 7 years. The best AI features I've shipped followed the opposite order:\n\nData first. Then context. Then model. Then evals. Then iterate.\n\nIt's boring. It works.\n\n#ProductManagement #AIProductManagement #PMTools",

    "The hardest part of shipping AI features is not the AI.\n\nIt's the part that comes before it: deciding exactly what the feature should do, what it should not do, and what happens when it gets it wrong.\n\nMost AI demos skip this. They answer every question. They hallucinate gracefully. They're impressive.\n\nProduction features can't do that. They need guardrails. They need to know when to say \"I don't know.\" They need to handle edge cases that the demo never considered.\n\nThe gap between demo and production is about 90% requirements engineering and 10% prompt tuning.\n\nMost teams spend it the other way around.\n\n#ProductManagement #AIProductManagement #AIAdoption",

    "The PMs who win in 2026 aren't the ones with the best prompts.\n\nThey're the ones who understand that the prompt is just the surface. The real work is in the retrieval layer, the context window management, the feedback loop between model output and user correction.\n\nI've watched good PMs get AI features wrong because they optimised the prompt first.\n\nI've watched better PMs get them right because they built the data pipeline first.\n\nThe prompt is the last 10% of the work.\n\n#ProductManagement #AIProductManagement #ContextEngineering",

    "RAG chatbot at SoftwareOne: 20-30% conversion lift. 55% support query reduction.\n\nPeople ask me what the secret was. Was it the model? The architecture? The prompt?\n\nNo. The secret was the data we fed it.\n\nWe spent six months mapping the support ticket taxonomy. Categorised every query. Built a retrieval system that actually understood the difference between \"I need my bill\" and \"why is my bill wrong.\"\n\nThe model was just the final piece.\n\nThe data was the product.\n\n#ProductManagement #AIProductManagement #RAG",
]

SAHIL_LINKEDIN_INDIE = [
    "I've built more AI-assisted tools for my own life this year than for any product team I've worked with.\n\nA bedtime story generator that turns my kids' day into a story (their day out to \"the great park adventure\" by 7:30pm). A takeaway roulette for when nobody can decide what to order. A media manager that catalogues my own family videos. A personal management assistant that pulls together calendar, notes, and tasks. A PM toolkit I use across my own indie projects.\n\nNone of these are products. None will ship publicly.\n\nBut here's what they taught me about AI in product management: the gap between \"AI is impressive\" and \"AI is useful\" is almost entirely about context.\n\nThe bedtime story generator is the same Claude API as everyone else's. What makes it actually work is what gets fed into it: the kids' ages, what they did that day, characters they like, story length appropriate for their attention spans, ending pattern that helps them sleep.\n\nThe model is the engine. The context is the entire vehicle.\n\nMost PMs I talk to are still tuning the engine.\n\n#ProductManagement #AIProductManagement #ContextEngineering #IndieDev",

    "I went from product management to indie building. The hardest part wasn't the code.\n\nIt was the context switching.\n\nAs a PM, you own one product. You know every detail. Every edge case. Every user frustration.\n\nAs an indie builder with four apps, you own nothing and everything. You're the PM, the developer, the designer, the support rep, and the person who fixes the server at midnight.\n\nThe thing that makes it work: ruthless prioritisation.\n\nNot every feature ships. Not every bug gets fixed today. Not every user request gets answered.\n\nShip the thing that moves the needle. Defer everything else.\n\n#ProductManagement #IndieHacker #BuildInPublic",

    "The indie portfolio strategy:\n\nBuild four things at once. Ship one. Learn from it. Ship the next. Repeat.\n\nKick-tionary went first. Tactics education for young players. It shipped because the scope was small.\n\nMatchdayMaestro is next. Football prediction app. 87% to launch. The UI is the blocker.\n\nPlenishd is after that. Kitchen inventory. Already got users testing.\n\nCoachOS is the furthest out. Grassroots coaching platform. August 2026 target.\n\nFour apps. One lesson each. That's the strategy.\n\n#ProductManagement #IndieHacker #BuildInPublic #PMTools",

    "My bedtime story generator has 3 user customisations.\n\nAll three are about pacing.\n\nOne parent wants 3-minute stories. One wants 8-minute stories. One wants \"as long as it takes — I need them asleep.\"\n\nThe model doesn't care about pacing. The context window does.\n\nThis is the pattern: user need maps to context, context maps to prompt, prompt maps to output.\n\nIf you skip the context layer, you don't have a product. You have a prompt.\n\n#ProductManagement #AIProductManagement #ContextEngineering",

    "What most people don't realise about indie building:\n\nShipping four apps at once isn't four times harder than shipping one. It's about twice harder.\n\nBecause the skills compound. The Convex schema you built for Plenishd works for CoachOS. The UI patterns you designed for MatchdayMaestro apply to Kick-tionary. The deployment pipeline you built once works for all of them.\n\nThe first app is the hardest. The second is easier. The third is where it starts feeling normal.\n\nStack your skills. Don't stack your problems.\n\n#ProductManagement #IndieHacker #BuildInPublic",
]

SAHIL_LINKEDIN_AI = [
    "Vibe coding makes you 3-4x faster at shipping.\n\nIt also makes you 10x faster at shipping vulnerabilities.\n\nVeracode tested AI-generated code across 80 tasks in 2026: 45% of samples failed security tests. 86% failed XSS defence. 88% were vulnerable to log injection.\n\nCodeRabbit's data shows AI-generated code introduces 2.74x more cross-site scripting vulnerabilities than human-written code.\n\nThe most uncomfortable part: this happens on apps that pass automated linters and standard test suites. The vulnerabilities aren't syntax errors — they're architectural choices the AI made that no static analyser is built to catch.\n\nI build apps as a solo developer and use AI heavily. This doesn't make me stop. It makes me more deliberate about three things:\n\n1. CI/CD that runs security scans before anything ships, not after.\n2. Manual audits before exposing anything to real users — the linter pass means nothing.\n3. Treating AI-generated code as a draft, not a deliverable.\n\nThe speed is real. The shortcut is fake.\n\n#VibeCoding #AIDevTools #ProductManagement #IndieHacker",

    "Georgia Tech's Vibe Security Radar tracked CVEs in AI-generated code:\n\nJanuary: 6 critical vulnerabilities found\nFebruary: 15\nMarch: 35\n\nThe trend line is not in the right direction.\n\nEscape.tech scanned 1,400+ vibe-coded production apps in 2026. 65% had security issues. 58% had at least one critical vulnerability. Over 400 exposed secrets.\n\nThe uncomfortable reality: most vibe-coders don't run security scans because the AI tool didn't tell them to. And the AI tool won't tell them to because the AI tool's job is shipping, not auditing.\n\nAI is the draft pen. Security review is still the human.\n\n#AIDevTools #VibeCoding #ProductManagement",

    "The gap between a Claude Code demo and a production AI feature is about six weeks.\n\nWeek 1: it works perfectly on the prompt you tested it with.\nWeek 2: it works on similar prompts, starts failing on edge cases.\nWeek 3: the edge cases multiply. You add guardrails.\nWeek 4: the guardrails break something else.\nWeek 5: you instrument it properly. You finally understand where it fails.\nWeek 6: it's production-ready. You're now 90% done.\n\nThe model was ready on Day 1. The system took six weeks.\n\nThat's the difference between an AI demo and a real feature.\n\n#AIProductManagement #ProductManagement #ContextEngineering",

    "Every data point I've seen in 2026 points to the same pattern:\n\nOpenRouter saw 1,236 vulnerabilities in their AI-assisted code samples.\nEscape found 654 exposed secrets in vibe-coded apps.\nGitHub Copilot users introduced 2.74x more XSS bugs than hand-written code.\n\nNot because AI is bad at security. Because AI optimises for working code, not secure code.\n\nThe fix: make security a first-class constraint in your AI coding workflow. Not a post-hoc audit. Not a \"we'll scan it before launch.\" A constraint in the prompt, in the CI/CD, in the review criteria.\n\nShip secure. Not fast and sorry.\n\n#VibeCoding #AIDevTools #ProductManagement #AIAdoption",

    "The most useful thing I learned building AI features this year:\n\nNever let the AI pick the architecture.\n\nAI-generated code passes the linter. It runs. It demos beautifully. The problems are always in the decisions it made silently: the way it structured the API response, the way it handled errors, the way it connected the database.\n\nHumans design the system. AI writes the plumbing. That's the boundary.\n\nCross it the other way and you're shipping vulnerabilities at 3x speed.\n\n#ProductManagement #AIProductManagement #AIDevTools",
]
SAHIL_LINKEDIN_LEADERSHIP = [
    "Enterprise AI adoption stalls at three points.\n\nAll of them are cultural, not technical.\n\n1. The 'who owns it' problem. No one wants to steward AI features once the demo is over. Engineers build it, PMs scope it, and then it sits in the backlog because nobody's performance metric is 'make AI work.'\n\n2. The context gap. Teams treat AI output quality as a model problem. It's a data problem. The prompt, the retrieval pipeline, the guardrails, the feedback loop — these shape 80%+ of the output.\n\n3. The measurement problem. Teams ship AI features but don't instrument them. 'Users love it' isn't a metric. Conversion lift is.\n\nI've seen this pattern at three companies now. The fix is always the same: assign an owner, fix the context, measure the result.\n\n#ProductManagement #AIProductManagement #Leadership",

    "The senior PM skill nobody talks about: knowing when to kill a feature.\n\nNot when it's clearly broken. Not when users are screaming about it.\n\nWhen it's almost good. When it's one sprint away. When the team has invested three months.\n\nThat's when the decision actually matters.\n\nAt Kinexio, we killed a feature that was 90% done because the remaining 10% was a compliance nightmare. It would have shipped. It would have worked. But the risk profile was wrong.\n\nKilling things is part of the job. Most PMs don't want to admit that.\n\n#ProductManagement #ProductLeadership #PMCareer",

    "The best product managers I've worked with share one trait:\n\nThey don't manage the product. They manage the information.\n\nThe right people have the right context at the right time. Engineering knows why we're building it, not just what. Sales knows what's coming so they don't over-promise. Leadership knows the trade-offs, not just the roadmap.\n\nMost PMs think their job is writing user stories.\n\nThe best ones know their job is making sure nobody ships the wrong thing.\n\n#ProductManagement #ProductLeadership #TechProductManagement",

    "What hiring managers actually look for in senior PMs in 2026:\n\nNot the number of features shipped. The quality of the decisions behind them.\n\nNot the size of the team managed. The clarity of communication under ambiguity.\n\nNot the tools they've used. The judgment they've built.\n\nAI makes every PM faster. It doesn't make them better. The delta is still strategic thinking, stakeholder management, and knowing when to push back.\n\nThe PMs who thrive aren't the ones with the best prompt libraries. They're the ones who already knew what questions to ask.\n\n#ProductManagement #PMCareer #AIProductManagement",

    "Your best PMs don't need to be told how to prioritise.\n\nThey just need the right constraints.\n\nGive a senior PM revenue targets, user data, and technical constraints, and they'll figure out the rest. Give them a prioritisation framework and they'll work around it.\n\nThe difference between PMs who need managing and PMs who manage themselves:\n\nThe ones who manage themselves understand the business model, not just the backlog. They optimise for outcomes, not output. They kill features that don't move the metric.\n\nBuild the constraints. Trust the person.\n\n#ProductManagement #ProductLeadership #TechProductManagement",
]


# ──────────────────────────────────────────────────────────────────────
# SAHIL TWITTER — ACTIVITY-DRIVEN (variable substitution templates)
# Generated from real signals via activity_collector.py
# ──────────────────────────────────────────────────────────────────────

SAHIL_TWITTER_GITHUB_PUSH = [
    "Just pushed `{repo_name}`.\n\n{description}\n\n#buildinpublic",
    "Shipped {repo_name} this week. {description}.\n\nMore in the repo.\n\n#buildinpublic",
    "Friday push: `{repo_name}` — {description}.\n\n#buildinpublic",
    "Pushed {repo_name}. {description}.\n\nShip small, ship often, ship everything public.\n\n#buildinpublic",
]

SAHIL_TWITTER_HERMES_PR = [
    "Submitted `{pr_title}` upstream to {upstream_repo}.\n\nOpen source: you contribute, framework gets better, everyone wins.\n\n#buildinpublic",
    "Another upstream contribution to {upstream_repo}: `{pr_title}`.\n\nSolo dev doesn't mean only building your own stuff.\n\n#buildinpublic",
    "PR up: `{pr_title}` on {upstream_repo} — {status}.\n\n#buildinpublic",
]

SAHIL_TWITTER_SKILL = [
    "Added a new skill to my Hermes setup today: {skill_name}.\n\nMy personal AI workforce just got another capability.\n\n#buildinpublic #AITools",
    "New skill shipped: {skill_name}. One more tool in the agent toolkit.\n\n#buildinpublic",
]

SAHIL_TWITTER_RESEARCH_TOOL = [
    "Research digest flagged this today: {title}\n\n{summary}\n\nWorth a look if you're in this space.\n\n#AITools",
    "Interesting tool surfaced — {title}.\n\n{summary}\n\nWorth a deeper look.\n\n#AITools",
]

SAHIL_TWITTER_RESEARCH_SIGNAL = [
    "Data point worth noting: {title}\n\n{summary}\n\n#AITools",
]

SAHIL_TWITTER_GITRADAR = [
    "GitHub radar surfaced `{repo_name}` today.\n\n{finding}\n\n#buildinpublic",
    "Radar pick: `{repo_name}`.\n\n{finding}\n\n#buildinpublic",
]

SAHIL_TWITTER_ARCHITECTURE = [
    "Architecture thought: {topic_label}.\n\n{description}.\n\n#buildinpublic",
    "On the architecture front: {topic_label}. {description}.\n\n#buildinpublic",
]

# ──────────────────────────────────────────────────────────────────────
# SAHIL LINKEDIN — ACTIVITY-DRIVEN
# Setup-Evidence-Frame structure tying activity to PM thinking
# ──────────────────────────────────────────────────────────────────────

SAHIL_LINKEDIN_GITHUB_PUSH = [
    "Just shipped {repo_name}.\n\nHere's what it is: {description}.\n\nWhat's the product lesson? A solo developer ships faster than most teams because the feedback loop is direct — no handoffs, no approvals, no asynchronous reviews. The bottleneck is never the code. It's the context switching between shipping and thinking about what to ship next.\n\nBuilding in public means the shipping is the thinking.\n\n#ProductManagement #BuildInPublic",
    "Shipped {repo_name} this week.\n\n{description}\n\nAs a PM, I've learned to treat every shipped thing as a hypothesis. The build is the fast part. Learning what people actually use it for — that takes longer.\n\n#ProductManagement #BuildInPublic",
]

SAHIL_LINKEDIN_HERMES_PR = [
    "Open source contribution I'm proud of: `{pr_title}` on {upstream_repo}.\n\nWhat this taught me about product thinking: contributing upstream forces you to think about the API surface, not just your internal implementation. You can't paper over design decisions when the PR is going to someone else's codebase.\n\nIt's the same discipline that makes good product requirements: write for the consumer, not yourself.\n\n#ProductManagement #OpenSource",
]

SAHIL_LINKEDIN_RESEARCH_TOOL = [
    "An interesting tool crossed my research desk today: {title}.\n\nWhat caught my attention: {summary}\n\nI evaluate tools the same way I evaluate features: does it solve a real problem, or does it solve a problem I only discovered because the tool exists? Most AI tools fall into the second bucket.\n\nThe product instinct is knowing the difference.\n\n#ProductManagement #AIProductManagement",
]

SAHIL_LINKEDIN_RESEARCH_SIGNAL = [
    "Data point worth pausing on: {title}\n\n{summary}\n\nNumbers like this shape how I think about product strategy — what to invest in, what to deprioritise, what to watch.\n\n#ProductManagement #AIProductManagement",
]

SAHIL_LINKEDIN_ARCHITECTURE = [
    "Architecture decision I made this week: {topic_label}.\n\nHere's the problem I was solving: {description}.\n\nWhy this is a product decision, not just an engineering one: how you structure your system determines what you can ship, how fast, and to whom. The architecture is the roadmap.\n\n#ProductManagement #AIProductManagement",
]

# Map signal_type → (twitter_templates, linkedin_templates)
ACTIVITY_TEMPLATES = {
    "github_push": (SAHIL_TWITTER_GITHUB_PUSH, SAHIL_LINKEDIN_GITHUB_PUSH),
    "hermes_pr": (SAHIL_TWITTER_HERMES_PR, SAHIL_LINKEDIN_HERMES_PR),
    "hermes_skill": (SAHIL_TWITTER_SKILL, None),
    "research_tool": (SAHIL_TWITTER_RESEARCH_TOOL, SAHIL_LINKEDIN_RESEARCH_TOOL),
    "research_signal": (SAHIL_TWITTER_RESEARCH_SIGNAL, SAHIL_LINKEDIN_RESEARCH_SIGNAL),
    "gitradar_repo": (SAHIL_TWITTER_GITRADAR, None),
    "architecture": (SAHIL_TWITTER_ARCHITECTURE, SAHIL_LINKEDIN_ARCHITECTURE),
}


def _fill_variables(template: str, variables: dict) -> Optional[str]:
    """Safely fill template variables. Returns None if required var missing."""
    try:
        result = template
        for key, val in variables.items():
            result = result.replace("{" + key + "}", str(val))
        # Check for unfilled placeholders
        unfilled = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", result)
        if unfilled:
            return None
        return result
    except (KeyError, ValueError, TypeError):
        return None


def _build_topic_variables(brand: str, topic: dict) -> dict:
    """Build a variable dict from topic data for template injection."""
    variables = {
        "topic": topic.get("topic", ""),
        "pillar": topic.get("pillar", ""),
        "brand": brand,
    }
    # MatchdayMaestro fixture injection
    fixture = topic.get("fixture")
    if fixture:
        variables["home"] = fixture.get("home", "Home")
        variables["away"] = fixture.get("away", "Away")
        variables["date"] = fixture.get("date", "")
        variables["time"] = fixture.get("time", "")
    # Activity-driven injection
    activity = topic.get("activity_data")
    if activity:
        variables.update(activity.get("variables", {}))
        variables["signal_type"] = activity.get("signal_type", "")
        variables["signal_id"] = activity.get("signal_id", "")
    return variables


def _is_future_fixture(topic: dict) -> bool:
    """Check if a topic references a fixture that is today or in the future."""
    fixture = topic.get("fixture")
    if not fixture:
        return True
    date_str = fixture.get("date", "")
    if not date_str:
        return True
    try:
        match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return match_date >= datetime.now(UTC).date()
    except ValueError:
        return True


def _reject_stale_tech(body: str) -> tuple[bool, str]:
    """Auto-reject drafts containing outdated AI tech references."""
    from topics import has_stale_tech_reference
    has_stale, match = has_stale_tech_reference(body)
    return not has_stale, match


ACTIVITY_SLOP_PATTERNS = [
    "What this tells me:",
    "[PM implication]",
    "[lesson]",
    "[reflection",
    "[insert",
]


def _has_unfilled_placeholders(body: str) -> bool:
    """Check if filled template still has unfilled placeholder patterns."""
    for pattern in ACTIVITY_SLOP_PATTERNS:
        if pattern in body:
            return True
    return False


COACHOS_SESSION_PLAN = [
    "8:45am. Sunday.\n\nPitch is playable but soft. Four players texted they're running late. One forgot boots.\nThe opposition's coach is already setting up cones on the far side.\n\nYou've got a session plan you wrote at 11pm on Friday. It's in a notes app somewhere. Or was it a printed card? Either way, the warm-up needs adapting -- two of your usual starters are the ones running late.\n\nGrassroots coaching. No two Sundays are the same.\n\nCoachOS helps with the plan, the adaptation, and the post-session debrief. The late players and the forgotten boots are between you and the group chat.\n\nBuilt for the coach on the sideline.",

    "You've got session plans in a notebook. Attendance in a WhatsApp poll. Player development notes scattered across your phone's notes app. Next week's fixture buried in a league email.\n\nAt the end of the season, a parent asks how their child has developed. You know the answer -- you've watched them for 30 sessions. But can you show it?\n\nCoachOS brings session planning, player tracking, team management, and comms into one place.\n\nPlan. Track. Improve.",

    "Most session plans fall apart in the first five minutes.\n\nNot because the plan was bad. Because four players are away, the pitch is smaller, and your holding midfielder -- the one the whole session was built around -- just texted he is ill.\n\nGood coaching is adapting while keeping the learning objective intact.\n\nThree things that help:\n\n1. Plan in progressions, not rigid drills. Every activity needs a simpler and harder version.\n2. Know your coaching points, not just your setup. The coaching point travels when the activity changes.\n3. Build sessions around principles, not players. If your key player is away, the session still works.\n\nCoachOS lets you build adaptable sessions and capture post-session notes so next week is better.\n\nSee you on Sunday.",

    "Wednesday night under floodlights. Cold. The training session you planned for 90 minutes needs to run in 60 because half the pitch is booked by the senior team.\n\nHere is the thing about coaching: if you have planned with progressions, you are fine. If you have planned fixed drills, you are stuck.\n\nCoachOS builds sessions in progressions by default. Every activity has an easier version and a harder version ready. When circumstances change, the session still works.\n\nLess admin. More coaching.",

    "Most coaches spend 45 minutes planning a 90-minute session.\n\nThen 90 minutes running it. Then zero minutes writing down what happened.\n\nSo next week, they are planning from memory.\n\n'I think the rondo worked' is not a good basis for a progressive session.\n\nCoachOS changes the last part. Post-session debrief takes two minutes. Voice note, hands still in gloves. Next week's plan starts from last week's notes.\n\nYour clipboard, upgraded.",

    "Building a session curriculum for the season is not glamorous.\n\nBut it is the difference between coaches who repeat the same six drills for 30 weeks and coaches whose players actually improve.\n\nBlock periodisation for grassroots:\n\nPre-season: fitness and first touch\nBlock 1: passing and possession\nBlock 2: defending as a unit\nBlock 3: transitions\n\nCoachOS lets you plan at the season level and drill down to the individual session. The whole curriculum, connected.\n\nBuilt for the coach on the sideline.",

    "The 60-minute session plan template:\n\nWarm-up: 8 minutes. Ball mastery and dynamic stretching.\nMain activity: 35 minutes. Progression-based. Start simple, build complexity.\nGame situation: 12 minutes. Apply the learning in a match context.\nCool-down: 5 minutes. Reflection and feedback.\n\nThat is it. Not 12 drills. Not 47 variations. One structure, applied differently each week.\n\nCoachOS stores these as templates. Pull one up. Adapt the drill. Go.\n\nLess admin. More coaching.",
]


COACHOS_COACH_LIFE = [
    "Things every grassroots coach has said this season:\n\n'I will remember what I changed last week.'\n'I will write it down after the session.'\n'I will send the attendance poll after training.'\n\nNone of them happened. Because you are knackered. It is Sunday night. Monday is a work day.\n\nCoachOS automates the post-session admin. Attendance logged. Session notes filed from voice. Next week's plan auto-suggested from what worked.\n\nOne place. Connected.\n\nLess admin. More coaching.",

    "The clipboard at grassroots football is a symbol of a bigger problem.\n\nIt is not the clipboard itself. It is the problem of information being trapped in a single format with no way to share, search, or review it later.\n\nSession plans on paper. Attendance on WhatsApp. Player development in your head. Fixtures in a league email.\n\nCoachOS puts it all in one place. Not as separate apps bolted together. As a connected system where attendance feeds into development, development shapes sessions, and sessions build into a season curriculum.\n\nYour clipboard, upgraded.",

    "The parent who asks after every session: 'How did they do today?'\n\nAnd the parent who never asks and assumes everything is fine.\n\nBoth deserve better than silence.\n\nCoachOS lets you share session summaries and individual player progress with parents. One tap. No individual messages needed. No group chat drama.\n\nPlan. Track. Share. Improve.",

    "Three seasons into grassroots, you have developed a system.\n\nSort the bibs by colour. Check the ball bag count. Lay out cones in the boot so the warm-up set is on top.\n\nThe experienced part is the instinct. The admin part is still a mess.\n\nCoachOS is for the coach who knows football but dreads the paperwork. Session planning, attendance tracking, player notes -- all automated, all connected.\n\nBuilt for the coach on the sideline.",

    "A coach's kit bag says everything:\n\nCones (some broken). Bibs (mismatched). A battered clipboard with three seasons of session plans in a plastic folder.\n\nThe clipboard has your history. Three years of drills, formations, warm-up routines. All handwritten. All impossible to search.\n\nCoachOS digitises the whole thing without losing the coaching instinct that made you good.\n\nPlan. Track. Improve. Your clipboard, upgraded.",

    "The end-of-season conversation every coach knows is coming.\n\nParent: 'How has my kid done this year?'\nYou: trying to remember 30 individual sessions from a notebook.\n\nThis is the moment where coaching admin matters most. Not at the planning stage. At the review stage.\n\nCoachOS tracks attendance, session notes, and individual player development across the whole season. When a parent asks, you show them. Not guess.\n\nBuilt for the coach on the sideline.",

    "Managing a grassroots team means managing expectations.\n\nThe parent who wants their kid starting. The volunteer assistant who misses half the sessions. The league secretary whose emails vanish into the void.\n\nCoachOS keeps everything in one place. Attendance visible to all. Session plans shared. Fixtures in one spot. No more 'I did not see the message.'\n\nLess chasing. More coaching.",
]


COACHOS_PLAYER_DEV = [
    "Most coaches know their players. But at the end of the season, knowing is not enough.\n\nYou watched them for 30 sessions. You saw the quiet kid find their confidence. You saw the natural striker learn to track back. You saw the goalkeeper who could not organise the defence start calling out positions.\n\nCan you show it to a parent? Can you show it to the player?\n\nCoachOS tracks individual development across the season. Session notes. Attendance. Progress markers. When a parent asks, you have the evidence.\n\nPlan. Track. Improve.",

    "The difference between coaching a session and managing a session:\n\nCoaching: actively teaching. Correcting technique. Building habits.\nManaging: just keeping them occupied for 90 minutes.\n\nMost grassroots coaches manage because they lack the tools. No session plans with clear learning objectives. No individual player development goals. No way to review what worked.\n\nCoachOS gives you all three. Structured plans. Individual tracking. Post-session review.\n\nLess managing, more coaching.",

    "Player development is slow. Invisible day by day. Obvious month by month.\n\nThe player who could not take a touch under pressure in September is doing it confidently by November. But did you notice? Can you prove it to their parent?\n\nCoachOS builds a development timeline for every player. Session notes. Attendance. Progress markers. Individual goals set and tracked.\n\nYour coaching instinct captures the moment. CoachOS captures the evidence.",

    "Age-appropriate coaching is where most coaches get it wrong.\n\nUnder 8s do not need 11-a-side tactics. They need ball mastery, spatial awareness, and fun. The session that works for a 15 year old will destroy a 7 year old's confidence.\n\nCoachOS organises session plans by age group and development stage. U7s are different from U12s, which are different from U16s. Not harder versions of the same thing. Fundamentally different approaches.\n\nPlan. Track. Improve.",

    "The hardest player development conversation is with the kid who thinks they are not good enough.\n\nThey train. They show up. But they do not see their own progress because they are comparing to the best player, not to the player they were three months ago.\n\nCoachOS tracks individual progress against individual baselines. Not the group. Their own starting point.\n\n'You could not take a touch under pressure in September. Now you can. Here is the evidence.'\n\nThat changes a kid's season.",

    "Session planning for player development is not about drills. It is about progressions.\n\nWeek 1: Introduction. Simple drill. Low complexity.\nWeek 2: Same concept, more pressure. Add defenders.\nWeek 3: Game situation. Apply the concept in a match context.\nWeek 4: Review. What worked? What needs reinforcement?\n\nCoachOS structures sessions as progressions, not one-offs. Each session builds on the last. Each review improves the next.\n\nPlayer development is not a lucky dip. It is a system.\n\nSee you on Sunday.",

    "Most grassroots clubs measure success by league position.\n\nBut league position hides everything:\n\nThe player who went from last picked to regular starter.\nThe team that went from panic to structure.\nThe coach who stopped shouting and started asking questions.\n\nCoachOS tracks the metrics that actually matter: attendance consistency, individual progress, session quality. Not just the score.\n\nBecause development wins in the long run. League tables lie in the short term.",
]


COACHOS_COMMUNITY = [
    "I have been coaching grassroots football for years. Sunday mornings, midweek training, cup runs and mid-table finishes. Still doing it.\n\nCoachOS is the tool I wish I had three seasons ago.\n\nWe are building it with input from coaches -- not in a survey, but in actual conversations about what makes Sunday mornings harder than they need to be.\n\nBeta opens August 2026. Pre-season. When you are actually planning.\n\nLess admin. More coaching.",

    "What do grassroots coaches actually need?\n\nWe asked them. Real conversations at training grounds and Sunday touchlines. Not a survey.\n\n'I need it to replace my clipboard, WhatsApp, and notes app.'\n'I need it to work when I am wearing gloves.'\n'I need it to show me what worked last week.'\n\nCoachOS is built from those conversations. Not assumptions.\n\nBuilt for the coach on the sideline.",

    "We are looking for grassroots coaches to shape CoachOS before beta.\n\nNot to test a finished product. To help build it from the ground up.\n\nTell us what is broken about your current coaching workflow. We will build the solution. You will test it. We will iterate. Together.\n\nBeta opens August 2026. The door is open.",

    "The pre-season window is the most important planning period for any grassroots coach.\n\nAugust is when you build your season curriculum. Set up sessions. Plan player development.\n\nCoachOS launches in August. Pre-season. When coaches are actually planning. Not January, when sessions are already in motion.\n\nThe timing is deliberate. Planning should start before the first whistle.\n\nPlan. Track. Improve. Less admin. More coaching.",

    "Meet the coaches shaping CoachOS:\n\nDave from Nottingham. U12s for three seasons. 'I do not want another app. I want everything in one place.'\nSarah from Leeds. Parent-coach. 'I need someone to tell me what a session plan actually looks like.'\nOmar from Birmingham. U16s. 'Player tracking is missing from every tool.'\n\nThree coaches. Three levels. Same core needs.\n\nCoachOS is being built for all of them.",

    "Pre-season starts in six weeks.\n\nDo you have your season curriculum mapped? Your individual player goals set? Your warm-up routines ready?\n\nOr are you winging it again?\n\nCoachOS helps you plan the whole season, not just next Saturday's drill. Block periodisation. Age-appropriate progressions. Post-season review. All connected.\n\nGet ahead of Sunday. Plan the season now.",
]


BRAND_TEMPLATES = {
    "matchdaymaestro": {
        "live_predictions": MM_LIVE_PREDICTIONS,
        "game_modes": MM_GAME_MODES,
        "friend_battles": MM_FRIEND_BATTLES,
        "progression": MM_PROGRESSION,
        "football_beat": MM_FOOTBALL_BEAT,
    },
    "plenishd": {
        "launch": PLENISHD_LAUNCH,
        "product": PLENISHD_PRODUCT,
        "voice": PLENISHD_VOICE,
        "kitchen": PLENISHD_KITCHEN,
        "shopping": PLENISHD_SHOPPING,
    },
    "sahil_twitter": {
        "build_in_public": SAHIL_TWITTER_BUILD,
        "ai_tools": SAHIL_TWITTER_BUILD,
        "sly_product": SAHIL_TWITTER_BUILD,
        "football": SAHIL_TWITTER_WRY,
        "wry": SAHIL_TWITTER_WRY,
        "tutorial": SAHIL_TWITTER_TUTORIAL,
        "data": SAHIL_TWITTER_DATA,
        "promotion": SAHIL_TWITTER_PROMOTION,
    },
    "sahil_linkedin": {
        "pm_thought": SAHIL_LINKEDIN_PM,
        "indie": SAHIL_LINKEDIN_INDIE,
        "ai": SAHIL_LINKEDIN_AI,
        "leadership": SAHIL_LINKEDIN_LEADERSHIP,
    },
    "coachos": {
        "session_plan": COACHOS_SESSION_PLAN,
        "coach_life": COACHOS_COACH_LIFE,
        "player_dev": COACHOS_PLAYER_DEV,
        "community": COACHOS_COMMUNITY,
    },
}


# _parse_llm_response removed -- template-only engine. LLM path was dead code.

def _audit_slop(body_text: str) -> dict:
    """Evaluate a draft body for AI-detectable patterns (slop_score).

    Returns dict with:
      - slop_score (0-10, higher = more detectable as AI-generated)
      - issues (list of specific problems found)
      - passed (bool: True if slop_score < 6)

    Pattern types detected:
    - Boilerplate mantras recurring verbatim across templates
    - Template-itis (identical sentence structure with one noun swapped)
    - Generic filler (non-specific, abstract language)
    - Repetitive hashtag patterns
    - Over-polished structure (setup -> tension -> resolution that's too clean)
    """
    issues = []
    slop_score = 0
    
    # Pattern 1: Boilerplate mantras (high weight)
    boilerplate_mantras = [
        "Shipping apps. Breaking things. Fixing them. Repeat.",
        "Tap to play.",
        "Make the call.",
        "Your move.",
        "Tomorrow's problem",
    ]
    for mantra in boilerplate_mantras:
        if mantra.lower() in body_text.lower():
            issues.append(f"Boilerplate mantra detected: '{mantra[:40]}'")
            slop_score += 2
    
    # Pattern 2: Template-itis — identical structure with one noun swapped
    # "Every AI tool now has a 'X' feature. Nobody solved why..."
    import re
    ai_tool_pattern = r"Every AI tool now has a '[\w\s]+' feature"
    if re.search(ai_tool_pattern, body_text):
        issues.append("Template-itis: 'Every AI tool now has a [X] feature' structure")
        slop_score += 2
    
    # Pattern 3: Generic filler — no specific names, numbers, or concrete details
    has_specific_number = bool(re.search(r'\b\d+%|\b\d+x|\b\d+ hours|\b£[\d.]+|\b\d+ items|\b\d+ apps|\b\d+ things\b', body_text))
    has_specific_tool = bool(re.search(r'(Claude|Convex|Supabase|Hermes|MCP|Postiz|Kinexio|SoftwareOne)', body_text, re.IGNORECASE))
    has_specific_app = bool(re.search(r'(Plenishd|MatchdayMaestro|CoachOS|Kick-tionary)', body_text, re.IGNORECASE))
    has_specific_time = bool(re.search(r'\b\d+ (weeks?|months?|days?|years?)\b', body_text, re.IGNORECASE))
    
    specificity_count = sum([has_specific_number, has_specific_tool, has_specific_app, has_specific_time])
    if specificity_count == 0:
        issues.append("No specific details (numbers, tools, apps, timeframes)")
        slop_score += 3
    elif specificity_count == 1:
        issues.append("Low specificity — only one concrete detail")
        slop_score += 1
    
    # Pattern 4: Over-polished structure detection
    # Triples with clean line breaks: short_line \n\n short_line \n\n short_line
    lines = [l.strip() for l in body_text.split('\n') if l.strip()]
    if len(lines) >= 4:
        clean_breaks = sum(1 for i in range(len(lines)) if len(lines[i]) < 30)
        if clean_breaks >= 3:
            issues.append("Over-polished short-line structure (AI-typical)")
            slop_score += 1
    
    return {
        "slop_score": min(slop_score, 10),
        "issues": issues,
        "passed": slop_score < 6,
    }


def _select_and_fill_template(brand: str, pillar: str, variables: dict) -> tuple[Optional[str], dict]:
    """Pick a template, inject variables, audit slop. Returns (body, audit);
    body is None when no candidate passed the audit."""
    templates = BRAND_TEMPLATES.get(brand, {}).get(pillar, [])
    if not templates:
        templates = [t for tlist in BRAND_TEMPLATES.get(brand, {}).values() for t in tlist]
    if not templates:
        templates = ["Draft coming soon."]

    max_attempts = min(len(templates), 5)
    body = None
    audit_result: dict = {"slop_score": 0, "issues": [], "passed": True}
    for _ in range(max_attempts):
        candidate = random.choice(templates)
        filled = _fill_variables(candidate, variables)
        if filled is None:
            continue
        audit_result = _audit_slop(filled)
        if audit_result["passed"]:
            body = filled
            break

    # Hard gate: nothing passed the slop audit, so nothing ships. A draft that
    # failed every attempt is exactly the output that reads as low quality.
    if body is None:
        audit_result["passed"] = False
        return None, audit_result

    return body, audit_result


def _generate_activity_draft(brand: str, topic: dict, platform: str, content_type: str) -> Optional[dict]:
    """Generate draft from activity signal with variable injection."""
    activity_data = topic.get("activity_data")
    if not activity_data:
        return None

    signal_type = activity_data.get("signal_type")
    variables = activity_data.get("variables", {})
    templates_tuple = ACTIVITY_TEMPLATES.get(signal_type)
    if not templates_tuple:
        return None

    is_linkedin = platform == "linkedin"
    idx = 1 if is_linkedin else 0
    templates = templates_tuple[idx]
    if not templates:
        return None

    body = None
    audit_result = None
    for _ in range(min(len(templates), 5)):
        candidate = random.choice(templates)
        filled = _fill_variables(candidate, variables)
        if filled is None or _has_unfilled_placeholders(filled):
            continue
        audit_result = _audit_slop(filled)
        if audit_result["passed"]:
            body = filled
            break

    if body is None:
        candidate = random.choice(templates)
        body = _fill_variables(candidate, variables) or candidate
        audit_result = _audit_slop(body)

    # Textless scene briefs only (no text/typography/UI -- the headline is a
    # Pillow overlay added later; see prompt_templates.build_scene_prompt).
    visual_descs = {
        "sahil_twitter": "a moody developer desk at night, warm screen glow, minimal",
        "sahil_linkedin": "a clean modern professional workspace, soft neutral light",
    }

    return {
        "body_text": body,
        "title": topic.get("topic", ""),
        "visual_description": visual_descs.get(brand, ""),
        "pillar": topic.get("pillar", ""),
        "topic": topic.get("topic", ""),
        "platform": platform,
        "content_type": content_type,
        "slop_audit": audit_result,
    }


def generate_drafts(brand, topics, platform=None, count_per_topic=1):
    """Generate drafts using brand voice templates with live variable injection.

    Template-only engine. LLM path removed (was dead code -- hermes -z timeouts).
    Every draft injects real topic data before template selection.
    Temporal gates: future fixtures only, stale tech auto-rejected.
    """
    drafts = []
    # Use real brand platform when none specified — fixes default-twitter bug
    from config import BRANDS
    brand_config = BRANDS.get(brand, {})
    brand_platforms = brand_config.get("platforms", [])
    if platform:
        platforms = [platform]
    elif brand_platforms:
        platforms = [brand_platforms[0]]
    else:
        platforms = ["twitter"]

    for topic in topics:
        for plat in platforms:
            pillar = topic.get("pillar", "")
            topic_text = topic.get("topic", "")
            content_type = _choose_content_type(brand, pillar, plat)

            # -- Temporal gate: skip past fixtures --
            if not _is_future_fixture(topic):
                continue

            # -- Build variable dict from topic data --
            variables = _build_topic_variables(brand, topic)

            # -- Activity-driven path (personal brands) --
            draft = None
            if brand in ("sahil_twitter", "sahil_linkedin") and topic.get("activity_data"):
                draft = _generate_activity_draft(brand, topic, plat, content_type)

            # -- sahil_twitter anti-scripted guardrail --
            # No static template fallback. Sahil personal X drafts MUST start
            # from a current internal activity/live signal. No activity_data →
            # no draft. Other brands continue through the static template path.
            if brand == "sahil_twitter" and draft is None:
                print(f"[llm_drafts] sahil_twitter/{pillar}: no activity signal; skipping (anti-scripted guard)")
                continue

            # -- Static template path (all brands except sahil_twitter) --
            if draft is None:
                body, audit_result = _select_and_fill_template(brand, pillar, variables)
                if body is None:
                    print(f"[llm_drafts] {brand}/{pillar}: no template passed slop audit; skipping topic")
                    continue

                # -- Stale tech gate (hard: a stale draft never ships) --
                passed_gate, stale_match = _reject_stale_tech(body)
                if not passed_gate:
                    body2, audit2 = _select_and_fill_template(brand, pillar, variables)
                    passed2, _ = _reject_stale_tech(body2 or "")
                    if body2 and passed2:
                        body = body2
                        audit_result = audit2
                    else:
                        print(f"[llm_drafts] {brand}/{pillar}: stale tech '{stale_match}' unresolved; skipping topic")
                        continue

                # Textless scene briefs only (no text/typography/UI/branding --
                # the headline is added later as a crisp Pillow overlay).
                visual_descs = {
                    "matchdaymaestro": "a floodlit football stadium at night, dramatic atmosphere",
                    "plenishd": "a warm sunlit kitchen scene, fresh ingredients, appetising",
                    "sahil_twitter": "a moody developer desk at night, warm screen glow, minimal",
                    "sahil_linkedin": "a clean modern professional workspace, soft neutral light",
                    "coachos": "a grassroots football pitch on an overcast morning, players training",
                }

                draft = {
                    "body_text": body,
                    "title": topic_text,
                    "visual_description": visual_descs.get(brand, ""),
                    "pillar": pillar,
                    "topic": topic_text,
                    "platform": plat,
                    "content_type": content_type,
                    "slop_audit": audit_result,
                }

            draft_id = f"{brand[:4]}_{str(uuid.uuid4())[:8]}"
            draft["id"] = draft_id
            draft["brand"] = brand
            draft["pillar"] = pillar
            draft["topic"] = topic_text
            draft["platform"] = plat
            draft["content_type"] = content_type

            # ── Claim/provenance gate for app brands ──
            if brand in ("plenishd", "coachos", "matchdaymaestro"):
                try:
                    from content_router import gate_claims
                    passes, issues = gate_claims(draft)
                    if not passes:
                        print(f"[llm_drafts] {brand}/{plat}: claim gate blocked ("
                              f"{' | '.join(issues)})")
                        continue  # skip this draft — don't save
                    draft["claim_issues"] = issues
                except Exception as exc:
                    print(f"[llm_drafts] claim gate error: {exc}")

            drafts.append(draft)

    return drafts
