"""
Hermes Devour Engine - Python 包装器

通过 CLI 调用 Rust 实现的外部开源吞噬引擎
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any


class DevourEngine:
    """外部开源吞噬引擎 Python 接口"""
    
    def __init__(self, devour_bin: str = "devour"):
        """
        初始化吞噬引擎
        
        Args:
            devour_bin: devour 二进制文件路径，默认从 PATH 查找
        """
        self.devour_bin = devour_bin
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        解析单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析结果字典，包含 language, functions, classes, imports 等
        """
        result = subprocess.run(
            [self.devour_bin, "parse", "--file", file_path, "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    
    def scan_directory(self, dir_path: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        扫描目录
        
        Args:
            dir_path: 目录路径
            output_file: 可选的输出文件路径
            
        Returns:
            扫描结果列表
        """
        cmd = [self.devour_bin, "scan", "--dir", dir_path]
        if output_file:
            cmd.extend(["--output", output_file])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if output_file:
            with open(output_file, 'r') as f:
                return json.load(f)
        else:
            return json.loads(result.stdout)
    
    def extract_capabilities(
        self,
        file_path: str,
        min_lines: int = 5,
        max_lines: int = 200,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        抽取能力
        
        Args:
            file_path: 文件路径
            min_lines: 最小函数行数
            max_lines: 最大函数行数
            output_file: 可选的输出文件路径
            
        Returns:
            能力列表
        """
        cmd = [
            self.devour_bin, "extract",
            "--file", file_path,
            "--min-lines", str(min_lines),
            "--max-lines", str(max_lines)
        ]
        if output_file:
            cmd.extend(["--output", output_file])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if output_file:
            with open(output_file, 'r') as f:
                return json.load(f)
        else:
            # 从 stdout 解析 JSON（忽略 stderr 的进度信息）
            return json.loads(result.stdout)


# 便捷函数
def parse_file(file_path: str) -> Dict[str, Any]:
    """解析单个文件"""
    engine = DevourEngine()
    return engine.parse_file(file_path)


def scan_directory(dir_path: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """扫描目录"""
    engine = DevourEngine()
    return engine.scan_directory(dir_path, output_file)


def extract_capabilities(
    file_path: str,
    min_lines: int = 5,
    max_lines: int = 200,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """抽取能力"""
    engine = DevourEngine()
    return engine.extract_capabilities(file_path, min_lines, max_lines, output_file)


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python devour_engine.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("=" * 70)
    print("解析文件")
    print("=" * 70)
    result = parse_file(file_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("抽取能力")
    print("=" * 70)
    capabilities = extract_capabilities(file_path, min_lines=3)
    print(f"找到 {len(capabilities)} 个能力")
    for cap in capabilities:
        print(f"\n  - {cap['name']}: {cap['capability_type']}")
        print(f"    复杂度: {cap['complexity_score']}/10")
        print(f"    可复用性: {cap['reusability_score']}/10")
        if cap['dependencies']:
            print(f"    依赖: {', '.join(cap['dependencies'])}")
