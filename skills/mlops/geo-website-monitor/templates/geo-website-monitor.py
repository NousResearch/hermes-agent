#!/usr/bin/env python3
"""
GEO官网监测工具 - 主程序脚本

功能：
1. 技术审计：检查网站的AI爬虫友好度
2. 内容友好度评估：基于GEO论文的9种优化方法
3. 引用测试：由Hermes Agent发送场景提示词做真实监测，并把结果回填到报告
4. 生成优化建议报告

使用方法：
    python geo-website-monitor.py audit https://example.com
    python geo-website-monitor.py content https://example.com
    python geo-website-monitor.py full https://example.com --scenarios scenarios.csv

设计原则：
- 每一个结论都有明确来源
- 不要提供主观的预期提升数据
- 引用论文时明确标注出处
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# GEO论文信息
GEO_PAPER_INFO = {
    "title": "GEO: Generative Engine Optimization",
    "authors": "Aggarwal, P., Murahari, V. S., Rajpurohit, S., Kalyan, A., Narasimhan, K., & Deshpande, A.",
    "conference": "KDD 2024",
    "arxiv": "2311.09735",
    "url": "https://generative-engines.com/GEO/"
}

# 基于论文的9种优化方法
GEO_METHODS = [
    {"id": "quotation_addition", "name": "Quotation Addition", "description": "添加可引用的直接引语", "source": "GEO论文表2"},
    {"id": "statistics_addition", "name": "Statistics Addition", "description": "添加具体数字、百分比", "source": "GEO论文表2"},
    {"id": "cite_sources", "name": "Cite Sources", "description": "引用权威机构/来源", "source": "GEO论文表2"},
    {"id": "fluency_optimization", "name": "Fluency Optimization", "description": "改进语法、表达", "source": "GEO论文表2"},
    {"id": "authoritative_tone", "name": "Authoritative Tone", "description": "更自信、权威的语气", "source": "GEO论文表2"},
    {"id": "easy_to_understand", "name": "Easy-to-Understand", "description": "更易理解的表达", "source": "GEO论文表2"},
    {"id": "unique_words", "name": "Unique Words", "description": "添加领域特有术语", "source": "GEO论文表2"},
    {"id": "technical_terms", "name": "Technical Terms", "description": "加入行业技术词汇", "source": "GEO论文表2"},
    {"id": "keyword_stuffing", "name": "Keyword Stuffing", "description": "传统SEO方法（不推荐）", "source": "GEO论文表2"}
]


@dataclass
class TechnicalAuditResult:
    """技术审计结果"""
    url: str
    https_ok: bool
    https_source: str
    robots_txt_exists: bool
    robots_txt_source: str
    robots_txt_allows_ai: bool
    robots_txt_ai_source: str
    llms_txt_exists: bool
    llms_txt_source: str
    login_wall: bool
    login_wall_source: str
    paywall: bool
    paywall_source: str
    core_content_in_html: bool
    core_content_source: str
    title_tag: bool
    title_source: str
    meta_keywords: bool
    meta_keywords_source: str
    meta_description: bool
    meta_description_source: str
    schema_org: bool
    schema_source: str
    opengraph: bool
    opengraph_source: str


@dataclass
class ContentScoreResult:
    """内容友好度评估结果"""
    url: str
    quotation_potential: int
    quotation_source: str
    data_potential: int
    data_source: str
    authority_potential: int
    authority_source: str
    fluency_score: int
    fluency_source: str
    authoritative_tone: int
    authoritative_tone_source: str
    readability: int
    readability_source: str
    unique_terms: int
    unique_terms_source: str
    technical_terms: int
    technical_terms_source: str
    keyword_stuffing_risk: int
    keyword_stuffing_source: str


@dataclass
class CitationTestResult:
    """引用测试结果"""
    scenario: str
    cited: bool
    cited_source: str
    mentioned: bool
    mentioned_source: str
    position: Optional[int]
    position_source: str
    model_answer: str
    cited_urls: List[str]


class GEOWebsiteMonitor:
    """GEO官网监测主类"""

    def __init__(self, url: str, brand_name: Optional[str] = None):
        self.url = url
        self.parsed_url = urlparse(url)
        self.brand_name = brand_name or self._extract_brand_name()
        self.html_content: Optional[str] = None
        self.fetch_source: Optional[str] = None

    def _extract_brand_name(self) -> str:
        """从URL中提取品牌名"""
        domain = self.parsed_url.netloc
        parts = domain.replace("www.", "").split(".")
        return parts[0].title() if parts else "Unknown"

    def fetch_website(self) -> Optional[str]:
        """获取网站内容"""
        try:
            # 先试HTTPS，再试HTTP
            for protocol in ["https", "http"]:
                try:
                    url = self.url.replace("https://", f"{protocol}://").replace("http://", f"{protocol}://")
                    if not url.startswith(protocol):
                        url = f"{protocol}://{url}"
                    response = requests.get(url, timeout=15, allow_redirects=True)
                    response.raise_for_status()
                    self.html_content = response.text
                    self.fetch_source = f"HTTP请求 {url} 返回状态码 {response.status_code}"
                    return self.html_content
                except requests.exceptions.RequestException:
                    continue
            self.fetch_source = "所有协议尝试失败"
            return None
        except Exception as e:
            self.fetch_source = f"获取网站异常: {e}"
            return None

    def run_technical_audit(self) -> TechnicalAuditResult:
        """运行技术审计"""
        if self.html_content is None:
            self.fetch_website()

        if self.html_content is None:
            raise Exception(f"无法获取网站内容，来源: {self.fetch_source}")

        soup = BeautifulSoup(self.html_content, "html.parser")

        # 检查HTTPS
        https_ok = self.url.startswith("https://")
        https_source = f"URL协议检查: {self.url}"

        # 检查robots.txt
        robots_txt_exists = False
        robots_txt_source = ""
        robots_txt_allows_ai = True
        robots_txt_ai_source = ""
        try:
            robots_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/robots.txt"
            robots_response = requests.get(robots_url, timeout=10)
            content = robots_response.text.lower()
            # 检查内容是否真的是robots.txt（不是返回200的404页面）
            if "<html" not in content and ("user-agent" in content or "disallow" in content or "allow" in content):
                robots_txt_exists = True
                robots_txt_source = f"HTTP请求 {robots_url} 返回状态码 {robots_response.status_code}，内容包含robots.txt关键词"
                # 检查是否禁止AI爬虫
                if any(bot in content for bot in ["gptbot", "claudebot", "google-extended", "baiduspider"]):
                    # 简单检查Disallow规则
                    robots_txt_allows_ai = not any(f"disallow: /" in line for line in content.split("\n") if "gptbot" in line)
                    robots_txt_ai_source = f"从robots.txt内容分析AI访问权限"
                else:
                    robots_txt_ai_source = "robots.txt中未找到针对AI爬虫的特殊规则，默认允许"
            else:
                robots_txt_source = f"HTTP请求 {robots_url} 返回状态码 {robots_response.status_code}，但内容是HTML或不包含robots.txt关键词"
        except Exception as e:
            robots_txt_source = f"HTTP请求 {robots_url} 异常: {e}"
            robots_txt_ai_source = "无法判断，因为robots.txt获取失败"

        # 检查llms.txt
        llms_txt_exists = False
        llms_txt_source = ""
        try:
            llms_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/llms.txt"
            llms_response = requests.get(llms_url, timeout=10)
            content = llms_response.text.lower()
            # 检查内容是否真的是llms.txt（不是返回200的404页面）
            if "<html" not in content and ("llms.txt" in content or "name:" in content or "allow:" in content):
                llms_txt_exists = True
                llms_txt_source = f"HTTP请求 {llms_url} 返回状态码 {llms_response.status_code}，内容包含llms.txt关键词"
            else:
                llms_txt_source = f"HTTP请求 {llms_url} 返回状态码 {llms_response.status_code}，但内容是HTML或不包含llms.txt关键词"
        except Exception as e:
            llms_txt_source = f"HTTP请求 {llms_url} 异常: {e}"

        # 检查登录墙
        login_wall = "login" in self.html_content.lower() and "password" in self.html_content.lower()
        login_wall_source = f"HTML内容中同时包含'login'和'password': {login_wall}"

        # 检查付费墙
        paywall = "pay" in self.html_content.lower() or "subscribe" in self.html_content.lower()
        paywall_source = f"HTML内容中包含'pay'或'subscribe': {paywall}"

        # 检查核心内容是否在HTML中
        core_content_in_html = True
        noscript_tags = soup.find_all("noscript")
        if len(noscript_tags) > 0:
            for noscript in noscript_tags:
                text = noscript.get_text().lower()
                if "javascript" in text or "enable" in text:
                    core_content_in_html = False
                    break
        core_content_source = f"检查noscript标签: 找到{len(noscript_tags)}个，其中提示需要JS: {not core_content_in_html}"

        # 检查标题标签
        title_tag = soup.title is not None and len(soup.title.text.strip()) > 0
        title_source = f"检查title标签: {'存在且非空' if title_tag else '不存在或为空'}"

        # 检查meta keywords
        meta_keywords = soup.find("meta", attrs={"name": "keywords"}) is not None
        meta_keywords_source = f"检查meta keywords: {'存在' if meta_keywords else '不存在'}"

        # 检查meta description
        meta_description = soup.find("meta", attrs={"name": "description"}) is not None
        meta_description_source = f"检查meta description: {'存在' if meta_description else '不存在'}"

        # 检查Schema.org
        schema_org = soup.find("script", attrs={"type": "application/ld+json"}) is not None
        schema_source = f"检查Schema.org (application/ld+json): {'存在' if schema_org else '不存在'}"

        # 检查OpenGraph
        opengraph = soup.find("meta", attrs={"property": "og:title"}) is not None
        opengraph_source = f"检查OpenGraph (og:title): {'存在' if opengraph else '不存在'}"

        return TechnicalAuditResult(
            url=self.url,
            https_ok=https_ok,
            https_source=https_source,
            robots_txt_exists=robots_txt_exists,
            robots_txt_source=robots_txt_source,
            robots_txt_allows_ai=robots_txt_allows_ai,
            robots_txt_ai_source=robots_txt_ai_source,
            llms_txt_exists=llms_txt_exists,
            llms_txt_source=llms_txt_source,
            login_wall=login_wall,
            login_wall_source=login_wall_source,
            paywall=paywall,
            paywall_source=paywall_source,
            core_content_in_html=core_content_in_html,
            core_content_source=core_content_source,
            title_tag=title_tag,
            title_source=title_source,
            meta_keywords=meta_keywords,
            meta_keywords_source=meta_keywords_source,
            meta_description=meta_description,
            meta_description_source=meta_description_source,
            schema_org=schema_org,
            schema_source=schema_source,
            opengraph=opengraph,
            opengraph_source=opengraph_source,
        )

    def run_content_score(self) -> ContentScoreResult:
        """运行内容友好度评估"""
        if self.html_content is None:
            self.fetch_website()

        if self.html_content is None:
            raise Exception(f"无法获取网站内容，来源: {self.fetch_source}")

        soup = BeautifulSoup(self.html_content, "html.parser")
        text_content = soup.get_text().strip()

        # 1. Quotation Addition（添加引语潜力）
        quotation_potential = 3
        quotation_source = "默认值3/10"
        blockquote_tags = soup.find_all("blockquote")
        q_tags = soup.find_all("q")
        if len(blockquote_tags) > 0 or len(q_tags) > 0:
            quotation_potential = 7
            quotation_source = f"找到{len(blockquote_tags)}个blockquote标签，{len(q_tags)}个q标签"
        elif "ceo" in text_content.lower() or "创始人" in text_content:
            quotation_potential = 6
            quotation_source = "内容中包含'CEO'或'创始人'，可能有引语素材"

        # 2. Statistics Addition（添加数据潜力）
        data_potential = 3
        data_source = "默认值3/10"
        numbers = re.findall(r'\d+', text_content)
        percentages = re.findall(r'\d+%', text_content)
        if len(numbers) > 10:
            data_potential = 6
            data_source = f"内容中找到{len(numbers)}个数字"
        if len(percentages) > 0:
            data_potential = 8
            data_source = f"内容中找到{len(percentages)}个百分比"
        if "专利" in text_content or "认证" in text_content:
            data_potential = max(data_potential, 7)
            data_source = data_source + "，且内容中包含'专利'或'认证'"

        # 3. Cite Sources（引用权威潜力）
        authority_potential = 3
        authority_source = "默认值3/10"
        if any(keyword in text_content.lower() for keyword in ["iso", "ce", "认证", "权威"]):
            authority_potential = 6
            authority_source = "内容中包含'ISO'、'CE'、'认证'或'权威'等关键词"
        external_links = soup.find_all("a", href=lambda href: href and href.startswith("http"))
        if len(external_links) > 5:
            authority_potential = max(authority_potential, 7)
            authority_source = authority_source + f"，且找到{len(external_links)}个外部链接"

        # 4. Fluency Optimization（流畅度得分）
        fluency_score = 7
        fluency_source = "默认值7/10"
        paragraphs = soup.find_all("p")
        if len(paragraphs) > 0:
            avg_paragraph_length = sum(len(p.get_text()) for p in paragraphs) / len(paragraphs)
            if 50 < avg_paragraph_length < 200:
                fluency_score = 8
                fluency_source = f"找到{len(paragraphs)}个p标签，平均长度{avg_paragraph_length:.1f}字符（在50-200范围内）"

        # 5. Authoritative Tone（权威语气）
        authoritative_tone = 6
        authoritative_tone_source = "默认值6/10"
        if any(keyword in text_content for keyword in ["标杆", "领导", "领先", "第一", "专家"]):
            authoritative_tone = 8
            authoritative_tone_source = "内容中包含'标杆'、'领导'、'领先'、'第一'或'专家'等权威语气词"

        # 6. Easy-to-Understand（简单易懂）
        readability = 7
        readability_source = "默认值7/10"
        h1_tags = soup.find_all("h1")
        h2_tags = soup.find_all("h2")
        if len(h1_tags) > 0 and len(h2_tags) > 0:
            readability = 8
            readability_source = f"找到{len(h1_tags)}个h1标签，{len(h2_tags)}个h2标签，标题层级清晰"

        # 7. Unique Words（独特词汇）
        unique_terms = 5
        unique_terms_source = "默认值5/10"
        if self.brand_name.lower() in text_content.lower():
            unique_terms = 7
            unique_terms_source = f"内容中包含品牌名'{self.brand_name}'"

        # 8. Technical Terms（技术术语）
        technical_terms = 5
        technical_terms_source = "默认值5/10"
        if any(keyword in text_content for keyword in ["系统", "技术", "专业", "智能"]):
            technical_terms = 7
            technical_terms_source = "内容中包含'系统'、'技术'、'专业'或'智能'等技术相关词汇"

        # 9. Keyword Stuffing（关键词堆砌风险）
        keyword_stuffing_risk = 2
        keyword_stuffing_source = "默认值2/10（低风险）"
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords:
            keywords_content = meta_keywords.get("content", "")
            if len(keywords_content.split(",")) > 10:
                keyword_stuffing_risk = 5
                keyword_stuffing_source = f"meta keywords包含{len(keywords_content.split(','))}个关键词（超过10个）"

        return ContentScoreResult(
            url=self.url,
            quotation_potential=quotation_potential,
            quotation_source=quotation_source,
            data_potential=data_potential,
            data_source=data_source,
            authority_potential=authority_potential,
            authority_source=authority_source,
            fluency_score=fluency_score,
            fluency_source=fluency_source,
            authoritative_tone=authoritative_tone,
            authoritative_tone_source=authoritative_tone_source,
            readability=readability,
            readability_source=readability_source,
            unique_terms=unique_terms,
            unique_terms_source=unique_terms_source,
            technical_terms=technical_terms,
            technical_terms_source=technical_terms_source,
            keyword_stuffing_risk=keyword_stuffing_risk,
            keyword_stuffing_source=keyword_stuffing_source,
        )

    def run_citation_test(self, scenario: str) -> CitationTestResult:
        """生成Agent引用测试占位结果。

        本脚本不调用豆包或任何模型API；真实引用测试由Hermes Agent发送
        场景提示词完成，并通过 --citation-results JSON 回填最终报告。
        """
        return CitationTestResult(
            scenario=scenario,
            cited=False,
            cited_source="未执行：Agent模式需由Hermes Agent发送场景提示词后通过--citation-results回填",
            mentioned=False,
            mentioned_source="未执行：Agent模式需由Hermes Agent发送场景提示词后通过--citation-results回填",
            position=None,
            position_source="未执行：Agent模式需由Hermes Agent发送场景提示词后通过--citation-results回填",
            model_answer="未执行真实引用测试。请先由Hermes Agent发送场景提示词，并将结果保存为JSON后用--citation-results导入。",
            cited_urls=[],
        )

    def generate_recommendations(self, technical: TechnicalAuditResult, content: ContentScoreResult) -> List[Dict]:
        """生成优化建议"""
        recommendations = []

        # 基于GEO论文的9种方法优先级排序
        # 来源：GEO论文表2，9种优化方法排名

        # 1. Quotation Addition（排名第1）
        if content.quotation_potential <= 5:
            recommendations.append({
                "priority": 1,
                "method": "Quotation Addition",
                "method_source": "GEO论文表2，排名第1",
                "description": "在'关于我们'中添加CEO引语、客户评价",
                "description_source": "工具建议",
                "evidence_source": content.quotation_source
            })

        # 2. Statistics Addition（排名第2）
        if content.data_potential <= 5:
            recommendations.append({
                "priority": 2,
                "method": "Statistics Addition",
                "method_source": "GEO论文表2，排名第2",
                "description": "增加具体数据、百分比、统计信息",
                "description_source": "工具建议",
                "evidence_source": content.data_source
            })

        # 3. Cite Sources（排名第3）
        if content.authority_potential <= 5:
            recommendations.append({
                "priority": 3,
                "method": "Cite Sources",
                "method_source": "GEO论文表2，排名第3",
                "description": "在页面中引用行业报告、权威机构认证、媒体报道",
                "description_source": "工具建议",
                "evidence_source": content.authority_source
            })

        # 技术优化建议
        if not technical.robots_txt_exists:
            recommendations.append({
                "priority": 4,
                "method": "Add robots.txt",
                "method_source": "SEO最佳实践",
                "description": "创建robots.txt文件",
                "description_source": "工具建议",
                "evidence_source": technical.robots_txt_source
            })

        if not technical.llms_txt_exists:
            recommendations.append({
                "priority": 5,
                "method": "Add llms.txt",
                "method_source": "AI内容发现最佳实践",
                "description": "创建llms.txt（AI发现文件）",
                "description_source": "工具建议",
                "evidence_source": technical.llms_txt_source
            })

        if not technical.schema_org:
            recommendations.append({
                "priority": 6,
                "method": "Add Schema.org structured data",
                "method_source": "SEO最佳实践",
                "description": "添加Organization、Product等Schema标记",
                "description_source": "工具建议",
                "evidence_source": technical.schema_source
            })

        if not technical.https_ok:
            recommendations.append({
                "priority": 7,
                "method": "Enable HTTPS",
                "method_source": "Web安全最佳实践",
                "description": "配置HTTPS证书",
                "description_source": "工具建议",
                "evidence_source": technical.https_source
            })

        return recommendations

    def generate_full_report(self, scenarios: Optional[List[str]] = None, agent_citation_results: Optional[List[CitationTestResult]] = None) -> Dict:
        """生成完整报告"""
        print("=== GEO官网监测 ===")
        print(f"目标网站: {self.url}")
        print(f"品牌名: {self.brand_name}")
        print(f"论文依据: {GEO_PAPER_INFO['title']} ({GEO_PAPER_INFO['conference']})")
        print()

        # 技术审计
        print("1. 技术审计...")
        technical = self.run_technical_audit()
        print("   技术审计完成")
        print()

        # 内容评估
        print("2. 内容友好度评估...")
        content = self.run_content_score()
        print("   内容评估完成")
        print()

        # 引用测试（最多5个场景词）：真实测试由Hermes Agent完成并通过JSON回填
        citation_results = []
        if agent_citation_results:
            print("3. 引用测试（使用Hermes Agent真实监测结果）...")
            citation_results = agent_citation_results[:5]
            print(f"   已导入 {len(citation_results)} 条Agent引用测试结果")
            print()
        elif scenarios:
            print("3. 引用测试（待Agent执行）...")
            for scenario in scenarios[:5]:
                result = self.run_citation_test(scenario)
                citation_results.append(result)
            print("   已生成待Agent执行的引用测试占位结果")
            print()

        # 生成建议
        print("4. 生成优化建议...")
        recommendations = self.generate_recommendations(technical, content)
        print(f"   生成了 {len(recommendations)} 条建议")
        print()

        # 构建报告
        report = {
            "metadata": {
                "website": self.url,
                "brand": self.brand_name,
                "paper_info": GEO_PAPER_INFO,
                "tool_version": "2.0 (source-verified)"
            },
            "technical_audit": {
                "https_ok": technical.https_ok,
                "https_source": technical.https_source,
                "robots_txt_exists": technical.robots_txt_exists,
                "robots_txt_source": technical.robots_txt_source,
                "robots_txt_allows_ai": technical.robots_txt_allows_ai,
                "robots_txt_ai_source": technical.robots_txt_ai_source,
                "llms_txt_exists": technical.llms_txt_exists,
                "llms_txt_source": technical.llms_txt_source,
                "login_wall": technical.login_wall,
                "login_wall_source": technical.login_wall_source,
                "paywall": technical.paywall,
                "paywall_source": technical.paywall_source,
                "core_content_in_html": technical.core_content_in_html,
                "core_content_source": technical.core_content_source,
                "title_tag": technical.title_tag,
                "title_source": technical.title_source,
                "meta_keywords": technical.meta_keywords,
                "meta_keywords_source": technical.meta_keywords_source,
                "meta_description": technical.meta_description,
                "meta_description_source": technical.meta_description_source,
                "schema_org": technical.schema_org,
                "schema_source": technical.schema_source,
                "opengraph": technical.opengraph,
                "opengraph_source": technical.opengraph_source
            },
            "content_evaluation": {
                "methods_basis": "GEO论文表2，9种优化方法",
                "quotation_addition": {
                    "score": content.quotation_potential,
                    "source": content.quotation_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "quotation_addition")
                },
                "statistics_addition": {
                    "score": content.data_potential,
                    "source": content.data_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "statistics_addition")
                },
                "cite_sources": {
                    "score": content.authority_potential,
                    "source": content.authority_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "cite_sources")
                },
                "fluency_optimization": {
                    "score": content.fluency_score,
                    "source": content.fluency_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "fluency_optimization")
                },
                "authoritative_tone": {
                    "score": content.authoritative_tone,
                    "source": content.authoritative_tone_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "authoritative_tone")
                },
                "easy_to_understand": {
                    "score": content.readability,
                    "source": content.readability_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "easy_to_understand")
                },
                "unique_words": {
                    "score": content.unique_terms,
                    "source": content.unique_terms_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "unique_words")
                },
                "technical_terms": {
                    "score": content.technical_terms,
                    "source": content.technical_terms_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "technical_terms")
                },
                "keyword_stuffing": {
                    "risk_score": content.keyword_stuffing_risk,
                    "source": content.keyword_stuffing_source,
                    "method_info": next(m for m in GEO_METHODS if m["id"] == "keyword_stuffing")
                }
            },
            "citation_tests": [
                {
                    "scenario": r.scenario,
                    "cited": r.cited,
                    "cited_source": r.cited_source,
                    "mentioned": r.mentioned,
                    "mentioned_source": r.mentioned_source,
                    "position": r.position,
                    "position_source": r.position_source,
                    "model_answer": r.model_answer,
                    "cited_urls": r.cited_urls
                }
                for r in citation_results
            ] if citation_results else None,
            "citation_summary": {
                "total_tests": len(citation_results) if citation_results else 0,
                "cited_count": sum(1 for r in citation_results if r.cited) if citation_results else 0,
                "mentioned_count": sum(1 for r in citation_results if r.mentioned) if citation_results else 0,
                "citation_rate": f"{sum(1 for r in citation_results if r.cited)/len(citation_results)*100:.1f}%" if citation_results else "N/A"
            } if citation_results else None,
            "recommendations": recommendations
        }

        return report


def load_scenarios(csv_path: str, limit: int = 5) -> List[str]:
    """从CSV加载场景词，最多返回5条。"""
    limit = min(max(int(limit), 1), 5)
    scenarios = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if row and row[0].strip():
                scenarios.append(row[0].strip())
            if len(scenarios) >= limit:
                break
    return scenarios


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s\)\]\}，。；;、]+", text)
    return list(dict.fromkeys(urls))


def load_agent_citation_results(json_path: str, brand_name: str, target_url: str, limit: int = 5) -> List[CitationTestResult]:
    """加载Hermes Agent真实引用测试结果。"""
    limit = min(max(int(limit), 1), 5)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("citation_tests") or payload.get("results") or [] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("citation results JSON must be a list or an object with citation_tests/results list")
    target_domain = urlparse(target_url).netloc.replace("www.", "").lower()
    brand_lower = (brand_name or "").lower()
    results: List[CitationTestResult] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        scenario = str(row.get("scenario", "")).strip()
        answer = str(row.get("model_answer") or row.get("answer") or "")
        cited_urls = [str(u) for u in (row.get("cited_urls") or _extract_urls(answer))]
        cited = row.get("cited")
        if cited is None:
            cited = any(target_domain and target_domain in u.lower() for u in cited_urls) or (target_domain and target_domain in answer.lower())
        mentioned = row.get("mentioned")
        if mentioned is None:
            mentioned = bool(brand_lower and brand_lower in answer.lower())
        position = row.get("position")
        if position is None and mentioned:
            idx = answer.lower().find(brand_lower) if brand_lower else -1
            position = idx + 1 if idx >= 0 else None
        results.append(CitationTestResult(
            scenario=scenario,
            cited=bool(cited),
            cited_source=row.get("cited_source") or "Hermes Agent真实发送场景提示词后，根据目标官网URL/域名判断",
            mentioned=bool(mentioned),
            mentioned_source=row.get("mentioned_source") or "Hermes Agent真实发送场景提示词后，根据品牌名判断",
            position=position,
            position_source=row.get("position_source") or "Hermes Agent真实回答中的首次出现字符位置；未出现则为null",
            model_answer=answer,
            cited_urls=cited_urls,
        ))
    return results


def main():
    parser = argparse.ArgumentParser(description="GEO官网监测工具")
    parser.add_argument("mode", choices=["audit", "content", "full"], help="运行模式")
    parser.add_argument("url", help="目标网站URL")
    parser.add_argument("--scenarios", help="场景词CSV文件路径（仅full模式，最多读取5条）")
    parser.add_argument("--scenario-count", type=int, default=5, help="场景提示词测试数量，1-5，默认5。Hermes Agent触发skill前必须主动询问用户。")
    parser.add_argument("--citation-results", help="Hermes Agent真实发送场景提示词后的结果JSON，用于回填最终报告")
    parser.add_argument("--brand", help="品牌名（可选）")
    parser.add_argument("--output", help="输出JSON文件路径（可选）")

    args = parser.parse_args()

    monitor = GEOWebsiteMonitor(args.url, args.brand)

    if args.mode == "audit":
        # 仅技术审计
        result = monitor.run_technical_audit()
        report = {
            "metadata": {
                "website": args.url,
                "brand": monitor.brand_name,
                "paper_info": GEO_PAPER_INFO
            },
            "technical_audit": result.__dict__
        }

    elif args.mode == "content":
        # 仅内容评估
        result = monitor.run_content_score()
        report = {
            "metadata": {
                "website": args.url,
                "brand": monitor.brand_name,
                "paper_info": GEO_PAPER_INFO
            },
            "content_evaluation": result.__dict__
        }

    elif args.mode == "full":
        # 完整测试
        scenario_count = min(max(args.scenario_count, 1), 5)
        scenarios = load_scenarios(args.scenarios, scenario_count) if args.scenarios else None
        agent_citation_results = load_agent_citation_results(args.citation_results, monitor.brand_name, args.url, scenario_count) if args.citation_results else None
        report = monitor.generate_full_report(scenarios, agent_citation_results=agent_citation_results)

    # 输出JSON
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存到: {args.output}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
