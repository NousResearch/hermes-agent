"""
cosmos Remotion Integration - Programmatic Video Generation.

"Lights, Camera, React!"

Features:
- Automated Project Scaffolding
- Dynamic Props Injection (Text, Images, Data)
- "Shorts" Optimization (9:16)
- Trade Recap Generation
"""

import subprocess
import os
import json
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List

class RemotionVideoSkill:
    def __init__(self, workspace_path: str = "./remotion_workspace"):
        self.workspace = Path(workspace_path)
        # Ensure workspace exists
        if not self.workspace.exists():
            try:
                self.workspace.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Remotion: Could not create workspace: {e}")

    async def init_project(self):
        """Scaffold a new Remotion project if one doesn't exist."""
        if (self.workspace / "package.json").exists():
            logger.info("Remotion: Project already exists.")
            return

        logger.info("Remotion: Initializing new video project...")
        # In a real environment, we would run: npx create-remotion@latest .
        # Since we are an agent, creating a few key files is safer/faster for now.
        
        # 1. package.json
        pkg = {
            "name": "cosmos-video",
            "version": "1.0.0",
            "scripts": {
                "start": "remotion preview src/index.ts",
                "render": "remotion render src/index.ts"
            },
            "dependencies": {
                "remotion": "latest",
                "react": "^18.0.0",
                "react-dom": "^18.0.0"
            }
        }
        with open(self.workspace / "package.json", "w") as f:
            json.dump(pkg, f, indent=2)

        # 2. src dir
        (self.workspace / "src").mkdir(exist_ok=True)
        
        # 3. Root component
        root_code = """
import {Composition} from 'remotion';
import {TradeRecap} from './TradeRecap';

export const RemotionRoot: React.FC = () => {
    return (
        <>
            <Composition
                id="TradeRecap"
                component={TradeRecap}
                durationInFrames={300}
                fps={30}
                width={1080}
                height={1920} // Shorts Format
                defaultProps={{
                    title: "SOLANA BREAKOUT",
                    profit: "+$4,200",
                    ticker: "SOL"
                }}
            />
        </>
    );
};
        """
        with open(self.workspace / "src/index.ts", "w") as f:
            f.write(root_code)

    async def render_video(self, composition_id: str, props: Dict[str, Any], output_filename: str) -> str:
        """
        Render a remotion composition to an MP4 file.
        """
        output_path = self.workspace / "out" / output_filename
        output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"Remotion: Rendering {composition_id} to {output_path}")
        
        # Serialize props to JSON for CLI
        props_file = self.workspace / "render_props.json"
        with open(props_file, "w") as f:
            json.dump(props, f)
            
        # Command Construction
        # Assumes 'npm install' has been run by the user or agent previously
        cmd = [
            "npx", "remotion", "render", 
            f"src/index.ts:{composition_id}", 
            str(output_path), 
            "--props", str(props_file),
            "--gl", "angle" # Use Angle backend for better compat in some envs
        ]
        
        try:
            # We run this in the background / sub-task
            process = subprocess.Popen(
                cmd, 
                cwd=str(self.workspace), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                shell=True if os.name == 'nt' else False
            )
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.success(f"Remotion: Render complete -> {output_path}")
                return str(output_path)
            else:
                err_msg = stderr.decode()
                logger.error(f"Remotion: Render failed: {err_msg}")
                return f"Error: {err_msg[:200]}..."
        except Exception as e:
            logger.error(f"Remotion: Execution error: {e}")
            return str(e)

    async def generate_trade_recap(self, trade_data: Dict) -> str:
        """
        High-level helper to generate a 'Shorts' style recap video for a trade.
        """
        await self.init_project()
        
        props = {
            "title": f"WINNING TRADE: {trade_data.get('ticker', 'UNKNOWN')}",
            "profit": trade_data.get('pnl_str', '$0'),
            "roi": trade_data.get('roi_str', '0%'),
            "entry": trade_data.get('entry_price', '0'),
            "exit": trade_data.get('exit_price', '0'),
            "theme": "dark_green" if float(trade_data.get('pnl', 0)) > 0 else "dark_red"
        }
        
        filename = f"recap_{trade_data.get('ticker')}_{int(os.times().elapsed)}.mp4"
        return await self.render_video("TradeRecap", props, filename)

remotion_skill = RemotionVideoSkill()
