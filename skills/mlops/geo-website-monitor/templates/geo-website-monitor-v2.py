#!/usr/bin/env python3
"""
GEO官网监测工具 - 主程序脚本（v2，来源验证版）

功能：
1. 技术审计：检查网站的AI爬虫友好度
2. 内容友好度评估：基于GEO论文的9种优化方法
3. 引用测试：由Hermes Agent发送场景提示词做真实监测，并把结果回填到报告
4. 生成优化建议报告

使用方法：
    python geo-website-monitor-v2.py audit https://example.com
    python geo-website-monitor-v2.py content https://example.com
    python geo-website-monitor-v2.py full https://example.com --scenarios scenarios.csv --citation-results agent-citation-results.json

设计原则：
- 每一个结论都有明确来源
- 不提供主观的预期提升数据
- 引用论文时明确标注出处
"""

import argparse
import csv
import json
import re
import sys
import time
import subprocess
import tempfile
from datetime import datetime
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

# 基于论文的9种优化方法（中英文对照）
GEO_METHODS = [
    {"id": "quotation_addition", "name_cn": "引语添加", "name_en": "Quotation Addition", "description": "添加可引用的直接引语", "source": "GEO论文表2"},
    {"id": "statistics_addition", "name_cn": "数据添加", "name_en": "Statistics Addition", "description": "添加具体数字、百分比", "source": "GEO论文表2"},
    {"id": "cite_sources", "name_cn": "权威引用", "name_en": "Cite Sources", "description": "引用权威机构/来源", "source": "GEO论文表2"},
    {"id": "fluency_optimization", "name_cn": "流畅度优化", "name_en": "Fluency Optimization", "description": "改进语法、表达", "source": "GEO论文表2"},
    {"id": "authoritative_tone", "name_cn": "权威语气", "name_en": "Authoritative Tone", "description": "更自信、权威的语气", "source": "GEO论文表2"},
    {"id": "easy_to_understand", "name_cn": "简单易懂", "name_en": "Easy-to-Understand", "description": "更易理解的表达", "source": "GEO论文表2"},
    {"id": "unique_words", "name_cn": "独特词汇", "name_en": "Unique Words", "description": "添加领域特有术语", "source": "GEO论文表2"},
    {"id": "technical_terms", "name_cn": "技术术语", "name_en": "Technical Terms", "description": "加入行业技术词汇", "source": "GEO论文表2"},
    {"id": "keyword_stuffing", "name_cn": "关键词堆砌", "name_en": "Keyword Stuffing", "description": "传统SEO方法（不推荐）", "source": "GEO论文表2"}
]


@dataclass
class TechnicalAuditResult:
    """技术审计结果"""
    url: str
    # 1.1 技术基础设施
    https_ok: bool
    https_source: str
    robots_txt_exists: bool
    robots_txt_source: str
    robots_txt_allows_ai: bool
    robots_txt_ai_source: str
    sitemap_exists: bool
    sitemap_source: str
    server_response_ok: bool
    server_response_time: float
    server_response_source: str
    mobile_friendly: bool
    mobile_friendly_source: str
    # 1.2 结构化数据与Schema
    schema_exists: bool
    schema_source: str
    schema_organization: bool
    schema_organization_source: str
    schema_core_types: List[str]
    schema_core_types_source: str
    schema_key_properties: bool
    schema_key_properties_source: str
    schema_complete: bool
    schema_complete_source: str
    # 1.3 内容结构
    homepage_complete: bool
    homepage_complete_source: str
    about_page_exists: bool
    about_page_source: str
    service_pages_exist: bool
    service_pages_source: str
    case_pages_exist: bool
    case_pages_source: str
    knowledge_base_exists: bool
    knowledge_base_source: str
    faq_page_exists: bool
    faq_page_source: str
    key_info_in_html: bool
    key_info_in_html_source: str
    content_structure_good: bool
    content_structure_source: str
    # 1.4 AI专属优化
    llms_txt_exists: bool
    llms_txt_source: str
    robots_txt_allows_ai_bots: bool
    robots_txt_allows_ai_bots_source: str
    eeat_signals: bool
    eeat_signals_source: str
    # 1.5 基础SEO
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
                except requests.exceptions.RequestException as e:
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

        # ========== 1.1 技术基础设施 ==========
        # HTTPS
        https_ok = self.url.startswith("https://")
        https_source = f"URL协议检查: {self.url}"

        # robots.txt
        robots_txt_exists = False
        robots_txt_source = ""
        robots_txt_allows_ai = True
        robots_txt_ai_source = ""
        robots_txt_allows_ai_bots = False
        robots_txt_allows_ai_bots_source = ""
        try:
            robots_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/robots.txt"
            robots_response = requests.get(robots_url, timeout=10)
            content = robots_response.text.lower()
            # 检查内容是否真的是robots.txt（不是返回200的404页面）
            if "<html" not in content and ("user-agent" in content or "disallow" in content or "allow" in content):
                robots_txt_exists = True
                robots_txt_source = f"HTTP请求 {robots_url} 返回状态码 {robots_response.status_code}，内容包含robots.txt关键词"
                # 检查是否允许AI爬虫
                ai_bots = ["gptbot", "google-extended", "perplexitybot", "claude bot", "anthropic-ai"]
                allows_any_ai_bot = False
                for bot in ai_bots:
                    if bot in content:
                        # 检查该bot是否被disallow
                        lines = content.split('\n')
                        bot_section = False
                        for line in lines:
                            line = line.strip().lower()
                            if f"user-agent: {bot}" in line:
                                bot_section = True
                            elif bot_section and line.startswith("user-agent:"):
                                bot_section = False
                            elif bot_section and line.startswith("disallow: /"):
                                robots_txt_allows_ai = False
                            elif bot_section and line.startswith("allow: /"):
                                allows_any_ai_bot = True
                if allows_any_ai_bot:
                    robots_txt_allows_ai_bots = True
                    robots_txt_allows_ai_bots_source = f"robots.txt中明确允许了AI爬虫（GPTBot、Google-Extended等）"
                else:
                    robots_txt_allows_ai_bots_source = f"robots.txt中未明确允许AI爬虫"
                robots_txt_ai_source = f"从robots.txt内容分析AI访问权限"
            else:
                robots_txt_source = f"HTTP请求 {robots_url} 返回状态码 {robots_response.status_code}，但内容是HTML或不包含robots.txt关键词"
        except Exception as e:
            robots_txt_source = f"HTTP请求 {robots_url} 异常: {e}"
            robots_txt_ai_source = "无法判断，因为robots.txt获取失败"
            robots_txt_allows_ai_bots_source = "无法判断，因为robots.txt获取失败"

        # Sitemap
        sitemap_exists = False
        sitemap_source = ""
        try:
            # 尝试常见的sitemap位置
            sitemap_urls = [
                f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/sitemap.xml",
                f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/sitemap_index.xml",
                f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/sitemap"
            ]
            for sitemap_url in sitemap_urls:
                try:
                    response = requests.get(sitemap_url, timeout=10)
                    content = response.text.lower()
                    if response.status_code == 200 and ("<sitemap" in content or "<urlset" in content):
                        sitemap_exists = True
                        sitemap_source = f"找到sitemap: {sitemap_url}"
                        break
                except:
                    continue
            if not sitemap_exists:
                # 检查robots.txt中是否有Sitemap指令
                if robots_txt_exists:
                    try:
                        robots_content = requests.get(robots_url, timeout=10).text.lower()
                        if "sitemap:" in robots_content:
                            sitemap_exists = True
                            sitemap_source = f"robots.txt中包含Sitemap指令"
                    except:
                        pass
            if not sitemap_exists:
                sitemap_source = "未在常见位置找到sitemap.xml，且robots.txt中无Sitemap指令"
        except Exception as e:
            sitemap_source = f"检查sitemap异常: {e}"

        # 服务器响应
        server_response_ok = False
        server_response_time = 0.0
        server_response_source = ""
        try:
            start_time = time.time()
            response = requests.get(self.url, timeout=15, allow_redirects=True)
            server_response_time = time.time() - start_time
            server_response_ok = response.status_code == 200
            server_response_source = f"HTTP请求 {self.url} 返回状态码 {response.status_code}，响应时间 {server_response_time:.2f}秒"
        except Exception as e:
            server_response_source = f"HTTP请求异常: {e}"

        # 移动端适配
        mobile_friendly = False
        mobile_friendly_source = ""
        try:
            # 检查viewport meta标签
            viewport = soup.find("meta", attrs={"name": "viewport"})
            if viewport:
                mobile_friendly = True
                mobile_friendly_source = "找到viewport meta标签，说明有移动端适配"
            else:
                # 检查是否有响应式CSS类名
                responsive_keywords = ["responsive", "mobile", "viewport", "media query"]
                css_links = soup.find_all("link", attrs={"rel": "stylesheet"})
                has_responsive_css = False
                for link in css_links:
                    href = link.get("href", "")
                    if any(keyword in href.lower() for keyword in responsive_keywords):
                        has_responsive_css = True
                        break
                if has_responsive_css:
                    mobile_friendly = True
                    mobile_friendly_source = "找到响应式相关的CSS文件"
                else:
                    mobile_friendly_source = "未找到viewport meta标签或响应式CSS"
        except Exception as e:
            mobile_friendly_source = f"检查移动端适配异常: {e}"

        # ========== 1.2 结构化数据与Schema ==========
        schema_exists = False
        schema_source = ""
        schema_organization = False
        schema_organization_source = ""
        schema_core_types = []
        schema_core_types_source = ""
        schema_key_properties = False
        schema_key_properties_source = ""
        schema_complete = False
        schema_complete_source = ""
        
        try:
            # 查找JSON-LD格式的Schema
            json_ld_scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
            if json_ld_scripts:
                schema_exists = True
                schema_source = f"找到{len(json_ld_scripts)}个JSON-LD格式的Schema标记"
                
                for script in json_ld_scripts:
                    try:
                        content = script.string
                        if content:
                            # 尝试解析JSON
                            data = json.loads(content)
                            # 处理@graph情况
                            schemas = data.get("@graph", [data]) if isinstance(data, dict) else [data]
                            
                            for schema in schemas:
                                if isinstance(schema, dict):
                                    schema_type = schema.get("@type", "")
                                    if isinstance(schema_type, list):
                                        schema_core_types.extend(schema_type)
                                    else:
                                        schema_core_types.append(schema_type)
                                    
                                    # 检查是否有Organization
                                    if schema_type == "Organization" or (isinstance(schema_type, list) and "Organization" in schema_type):
                                        schema_organization = True
                                        schema_organization_source = "找到Organization类型的Schema"
                                    
                                    # 检查关键属性
                                    if "@id" in schema or "sameAs" in schema:
                                        schema_key_properties = True
                                        schema_key_properties_source = "Schema包含@id或sameAs关键属性"
                                    
                                    # 检查Schema完整性（是否有多个字段）
                                    if len(schema.keys()) > 5:
                                        schema_complete = True
                                        schema_complete_source = "Schema属性较完整（超过5个字段）"
                    except json.JSONDecodeError:
                        continue
                
                schema_core_types = list(set(schema_core_types))  # 去重
                if schema_core_types:
                    schema_core_types_source = f"找到的Schema类型: {', '.join(schema_core_types)}"
                else:
                    schema_core_types_source = "未识别到具体的Schema类型"
            else:
                schema_source = "未找到JSON-LD格式的Schema标记"
        except Exception as e:
            schema_source = f"检查Schema异常: {e}"

        # ========== 1.3 内容结构 ==========
        # 首页完整性（检查首页是否包含关键信息）
        homepage_complete = False
        homepage_complete_source = ""
        try:
            text = soup.get_text().lower()
            keywords = ["定位", "业务", "服务", "案例", "介绍", "关于"]
            found_keywords = [kw for kw in keywords if kw in text]
            if len(found_keywords) >= 3:
                homepage_complete = True
                homepage_complete_source = f"首页包含关键信息: {', '.join(found_keywords)}"
            else:
                homepage_complete_source = f"首页关键信息较少，仅找到: {', '.join(found_keywords) if found_keywords else '无'}"
        except Exception as e:
            homepage_complete_source = f"检查首页完整性异常: {e}"

        # 检查关键页面是否存在（通过链接检测）
        about_page_exists = False
        about_page_source = ""
        service_pages_exist = False
        service_pages_source = ""
        case_pages_exist = False
        case_pages_source = ""
        knowledge_base_exists = False
        knowledge_base_source = ""
        faq_page_exists = False
        faq_page_source = ""
        
        try:
            links = soup.find_all("a", href=True)
            link_texts = [a.get_text().lower() for a in links]
            link_hrefs = [a.get("href", "").lower() for a in links]
            
            # 关于我们
            about_keywords = ["关于", "about", "公司介绍", "关于我们"]
            for text in link_texts + link_hrefs:
                if any(kw in text for kw in about_keywords):
                    about_page_exists = True
                    about_page_source = f"找到关于我们相关链接: {text}"
                    break
            if not about_page_exists:
                about_page_source = "未找到关于我们相关链接"
            
            # 服务中心
            service_keywords = ["服务", "service", "产品", "解决方案"]
            for text in link_texts + link_hrefs:
                if any(kw in text for kw in service_keywords):
                    service_pages_exist = True
                    service_pages_source = f"找到服务相关链接: {text}"
                    break
            if not service_pages_exist:
                service_pages_source = "未找到服务相关链接"
            
            # 案例中心
            case_keywords = ["案例", "case", "客户", "成功案例"]
            for text in link_texts + link_hrefs:
                if any(kw in text for kw in case_keywords):
                    case_pages_exist = True
                    case_pages_source = f"找到案例相关链接: {text}"
                    break
            if not case_pages_exist:
                case_pages_source = "未找到案例相关链接"
            
            # 知识库/博客
            kb_keywords = ["知识", "博客", "blog", "文章", "资讯", "新闻"]
            for text in link_texts + link_hrefs:
                if any(kw in text for kw in kb_keywords):
                    knowledge_base_exists = True
                    knowledge_base_source = f"找到知识库/博客相关链接: {text}"
                    break
            if not knowledge_base_exists:
                knowledge_base_source = "未找到知识库/博客相关链接"
            
            # FAQ页面
            faq_keywords = ["faq", "常见问题", "问题解答", "问答"]
            for text in link_texts + link_hrefs:
                if any(kw in text for kw in faq_keywords):
                    faq_page_exists = True
                    faq_page_source = f"找到FAQ相关链接: {text}"
                    break
            if not faq_page_exists:
                faq_page_source = "未找到FAQ相关链接"
        except Exception as e:
            about_page_source = f"检查关于我们页面异常: {e}"
            service_pages_source = f"检查服务页面异常: {e}"
            case_pages_source = f"检查案例页面异常: {e}"
            knowledge_base_source = f"检查知识库页面异常: {e}"
            faq_page_source = f"检查FAQ页面异常: {e}"

        # 关键信息格式（检查是否只有图片没有文本）
        key_info_in_html = True
        key_info_in_html_source = ""
        try:
            images = soup.find_all("img")
            total_text_length = len(soup.get_text().strip())
            # 如果图片很多但文本很少，可能关键信息在图片中
            if len(images) > 10 and total_text_length < 500:
                key_info_in_html = False
                key_info_in_html_source = f"页面有{len(images)}张图片但文本仅{total_text_length}字，可能关键信息在图片中"
            else:
                key_info_in_html_source = f"页面有{len(images)}张图片和{total_text_length}字文本"
        except Exception as e:
            key_info_in_html_source = f"检查关键信息格式异常: {e}"

        # 内容结构（检查是否有"是什么、为什么、怎么做"等结构）
        content_structure_good = False
        content_structure_source = ""
        try:
            headings = []
            for h in ["h1", "h2", "h3"]:
                headings.extend([tag.get_text().lower() for tag in soup.find_all(h)])
            
            structure_keywords = ["是什么", "为什么", "怎么做", "如何", "常见问题", "注意事项"]
            found_structure = [kw for kw in structure_keywords if any(kw in h for h in headings)]
            
            if len(found_structure) >= 2:
                content_structure_good = True
                content_structure_source = f"内容结构良好，包含: {', '.join(found_structure)}"
            else:
                content_structure_source = f"内容结构一般，仅找到: {', '.join(found_structure) if found_structure else '无'}"
        except Exception as e:
            content_structure_source = f"检查内容结构异常: {e}"

        # ========== 1.4 AI专属优化 ==========
        # llms.txt（这个已经在上面检查过了，这里复用一下逻辑）
        if 'llms_txt_exists' not in locals():
            llms_txt_exists = False
            llms_txt_source = ""
            try:
                llms_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/llms.txt"
                llms_response = requests.get(llms_url, timeout=10)
                content = llms_response.text.lower()
                if "<html" not in content and ("llms.txt" in content or "name:" in content or "allow:" in content):
                    llms_txt_exists = True
                    llms_txt_source = f"HTTP请求 {llms_url} 返回状态码 {llms_response.status_code}，内容包含llms.txt关键词"
                else:
                    llms_txt_source = f"HTTP请求 {llms_url} 返回状态码 {llms_response.status_code}，但内容是HTML或不包含llms.txt关键词"
            except Exception as e:
                llms_txt_source = f"HTTP请求 {llms_url} 异常: {e}"

        # E-E-A-T信号
        eeat_signals = False
        eeat_signals_source = ""
        try:
            text = soup.get_text().lower()
            eeat_keywords = ["经验", "专业", "权威", "可信", "资质", "认证", "专家", "团队", "背景"]
            found_eeat = [kw for kw in eeat_keywords if kw in text]
            if len(found_eeat) >= 3:
                eeat_signals = True
                eeat_signals_source = f"内容包含E-E-A-T信号: {', '.join(found_eeat)}"
            else:
                eeat_signals_source = f"E-E-A-T信号较少，仅找到: {', '.join(found_eeat) if found_eeat else '无'}"
        except Exception as e:
            eeat_signals_source = f"检查E-E-A-T信号异常: {e}"

        # ========== 1.5 基础SEO ==========
        # 登录墙
        login_wall = "login" in self.html_content.lower() and "password" in self.html_content.lower()
        login_wall_source = f"HTML内容中同时包含'login'和'password': {login_wall}"

        # 付费墙
        paywall = "pay" in self.html_content.lower() or "subscribe" in self.html_content.lower()
        paywall_source = f"HTML内容中包含'pay'或'subscribe': {paywall}"

        # 核心内容是否在HTML中
        core_content_in_html = True
        noscript_tags = soup.find_all("noscript")
        if len(noscript_tags) > 0:
            for noscript in noscript_tags:
                noscript_text = noscript.get_text().lower()
                if "javascript" in noscript_text or "enable" in noscript_text:
                    core_content_in_html = False
                    break
        core_content_source = f"检查noscript标签: 找到{len(noscript_tags)}个，其中提示需要JS: {not core_content_in_html}"

        # 标题标签
        title_tag = soup.title is not None and len(soup.title.text.strip()) > 0
        title_source = f"检查title标签: {'存在且非空' if title_tag else '不存在或为空'}"

        # Meta Keywords
        meta_keywords = soup.find("meta", attrs={"name": "keywords"}) is not None
        meta_keywords_source = f"检查meta keywords: {'存在' if meta_keywords else '不存在'}"

        # Meta Description
        meta_description = soup.find("meta", attrs={"name": "description"}) is not None
        meta_description_source = f"检查meta description: {'存在' if meta_description else '不存在'}"

        # OpenGraph
        opengraph = soup.find("meta", attrs={"property": "og:title"}) is not None
        opengraph_source = f"检查OpenGraph (og:title): {'存在' if opengraph else '不存在'}"

        return TechnicalAuditResult(
            url=self.url,
            # 1.1 技术基础设施
            https_ok=https_ok,
            https_source=https_source,
            robots_txt_exists=robots_txt_exists,
            robots_txt_source=robots_txt_source,
            robots_txt_allows_ai=robots_txt_allows_ai,
            robots_txt_ai_source=robots_txt_ai_source,
            sitemap_exists=sitemap_exists,
            sitemap_source=sitemap_source,
            server_response_ok=server_response_ok,
            server_response_time=server_response_time,
            server_response_source=server_response_source,
            mobile_friendly=mobile_friendly,
            mobile_friendly_source=mobile_friendly_source,
            # 1.2 结构化数据与Schema
            schema_exists=schema_exists,
            schema_source=schema_source,
            schema_organization=schema_organization,
            schema_organization_source=schema_organization_source,
            schema_core_types=schema_core_types,
            schema_core_types_source=schema_core_types_source,
            schema_key_properties=schema_key_properties,
            schema_key_properties_source=schema_key_properties_source,
            schema_complete=schema_complete,
            schema_complete_source=schema_complete_source,
            # 1.3 内容结构
            homepage_complete=homepage_complete,
            homepage_complete_source=homepage_complete_source,
            about_page_exists=about_page_exists,
            about_page_source=about_page_source,
            service_pages_exist=service_pages_exist,
            service_pages_source=service_pages_source,
            case_pages_exist=case_pages_exist,
            case_pages_source=case_pages_source,
            knowledge_base_exists=knowledge_base_exists,
            knowledge_base_source=knowledge_base_source,
            faq_page_exists=faq_page_exists,
            faq_page_source=faq_page_source,
            key_info_in_html=key_info_in_html,
            key_info_in_html_source=key_info_in_html_source,
            content_structure_good=content_structure_good,
            content_structure_source=content_structure_source,
            # 1.4 AI专属优化
            llms_txt_exists=llms_txt_exists,
            llms_txt_source=llms_txt_source,
            robots_txt_allows_ai_bots=robots_txt_allows_ai_bots,
            robots_txt_allows_ai_bots_source=robots_txt_allows_ai_bots_source,
            eeat_signals=eeat_signals,
            eeat_signals_source=eeat_signals_source,
            # 1.5 基础SEO
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
            opengraph=opengraph,
            opengraph_source=opengraph_source
        )

    def run_content_score(self) -> ContentScoreResult:
        """运行内容友好度评估"""
        if self.html_content is None:
            self.fetch_website()

        if self.html_content is None:
            raise Exception(f"无法获取网站内容，来源: {self.fetch_source}")

        soup = BeautifulSoup(self.html_content, "html.parser")
        text_content = soup.get_text().strip()

        # 1. Quotation Addition (添加引语潜力)
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

        # 2. Statistics Addition (添加数据潜力)
        data_potential = 3
        data_source = "默认值3/10"
        numbers = re.findall(r"\d+", text_content)
        percentages = re.findall(r"\d+%", text_content)
        if len(numbers) > 10:
            data_potential = 6
            data_source = f"内容中找到{len(numbers)}个数字"
        if len(percentages) > 0:
            data_potential = 8
            data_source = f"内容中找到{len(percentages)}个百分比"
        if "专利" in text_content or "认证" in text_content:
            data_potential = max(data_potential, 7)
            data_source = data_source + "，且内容中包含'专利'或'认证'"

        # 3. Cite Sources (引用权威潜力)
        authority_potential = 3
        authority_source = "默认值3/10"
        if any(keyword in text_content.lower() for keyword in ["iso", "ce", "认证", "权威"]):
            authority_potential = 6
            authority_source = "内容中包含'ISO'、'CE'、'认证'或'权威'等关键词"
        external_links = soup.find_all("a", href=lambda href: href and href.startswith("http"))
        if len(external_links) > 5:
            authority_potential = max(authority_potential, 7)
            authority_source = authority_source + f"，且找到{len(external_links)}个外部链接"

        # 4. Fluency Optimization (流畅度得分)
        fluency_score = 7
        fluency_source = "默认值7/10"
        paragraphs = soup.find_all("p")
        if len(paragraphs) > 0:
            avg_paragraph_length = sum(len(p.get_text()) for p in paragraphs) / len(paragraphs)
            if 50 < avg_paragraph_length < 200:
                fluency_score = 8
                fluency_source = f"找到{len(paragraphs)}个p标签，平均长度{avg_paragraph_length:.1f}字符（在50-200范围内）"

        # 5. Authoritative Tone (权威语气)
        authoritative_tone = 6
        authoritative_tone_source = "默认值6/10"
        if any(keyword in text_content for keyword in ["标杆", "领导", "领先", "第一", "专家"]):
            authoritative_tone = 8
            authoritative_tone_source = "内容中包含'标杆'、'领导'、'领先'、'第一'或'专家'等权威语气词"

        # 6. Easy-to-Understand (简单易懂)
        readability = 7
        readability_source = "默认值7/10"
        h1_tags = soup.find_all("h1")
        h2_tags = soup.find_all("h2")
        if len(h1_tags) > 0 and len(h2_tags) > 0:
            readability = 8
            readability_source = f"找到{len(h1_tags)}个h1标签，{len(h2_tags)}个h2标签，标题层级清晰"

        # 7. Unique Words (独特词汇)
        unique_terms = 5
        unique_terms_source = "默认值5/10"
        if self.brand_name.lower() in text_content.lower():
            unique_terms = 7
            unique_terms_source = f"内容中包含品牌名'{self.brand_name}'"

        # 8. Technical Terms (技术术语)
        technical_terms = 5
        technical_terms_source = "默认值5/10"
        if any(keyword in text_content for keyword in ["系统", "技术", "专业", "智能"]):
            technical_terms = 7
            technical_terms_source = "内容中包含'系统'、'技术'、'专业'或'智能'等技术相关词汇"

        # 9. Keyword Stuffing (关键词堆砌风险)
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

        注意：本脚本不调用豆包或任何模型API。真实引用测试必须由Hermes Agent
        将场景提示词发送给当前模型完成，然后通过 --citation-results JSON 回填。
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
        # 来源：GEO论文表2（9种优化方法排名）

        # 1. Quotation Addition (排名第1)
        if content.quotation_potential <= 5:
            recommendations.append({
                "priority": 1,
                "method": "Quotation Addition",
                "method_source": "GEO论文表2，排名第1",
                "description": "在'关于我们'中添加CEO引语、客户评价",
                "description_source": "工具建议",
                "evidence_source": content.quotation_source
            })

        # 2. Statistics Addition (排名第2)
        if content.data_potential <= 5:
            recommendations.append({
                "priority": 2,
                "method": "Statistics Addition",
                "method_source": "GEO论文表2，排名第2",
                "description": "增加具体数据、百分比、统计信息",
                "description_source": "工具建议",
                "evidence_source": content.data_source
            })

        # 3. Cite Sources (排名第3)
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

        if not technical.schema_exists:
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
                "tool_version": "3.0 (GEO白皮书优化版)"
            },
            "technical_audit": {
                # 1.1 技术基础设施
                "https_ok": technical.https_ok,
                "https_source": technical.https_source,
                "robots_txt_exists": technical.robots_txt_exists,
                "robots_txt_source": technical.robots_txt_source,
                "robots_txt_allows_ai": technical.robots_txt_allows_ai,
                "robots_txt_ai_source": technical.robots_txt_ai_source,
                "sitemap_exists": technical.sitemap_exists,
                "sitemap_source": technical.sitemap_source,
                "server_response_ok": technical.server_response_ok,
                "server_response_time": technical.server_response_time,
                "server_response_source": technical.server_response_source,
                "mobile_friendly": technical.mobile_friendly,
                "mobile_friendly_source": technical.mobile_friendly_source,
                # 1.2 结构化数据与Schema
                "schema_exists": technical.schema_exists,
                "schema_source": technical.schema_source,
                "schema_organization": technical.schema_organization,
                "schema_organization_source": technical.schema_organization_source,
                "schema_core_types": technical.schema_core_types,
                "schema_core_types_source": technical.schema_core_types_source,
                "schema_key_properties": technical.schema_key_properties,
                "schema_key_properties_source": technical.schema_key_properties_source,
                "schema_complete": technical.schema_complete,
                "schema_complete_source": technical.schema_complete_source,
                # 1.3 内容结构
                "homepage_complete": technical.homepage_complete,
                "homepage_complete_source": technical.homepage_complete_source,
                "about_page_exists": technical.about_page_exists,
                "about_page_source": technical.about_page_source,
                "service_pages_exist": technical.service_pages_exist,
                "service_pages_source": technical.service_pages_source,
                "case_pages_exist": technical.case_pages_exist,
                "case_pages_source": technical.case_pages_source,
                "knowledge_base_exists": technical.knowledge_base_exists,
                "knowledge_base_source": technical.knowledge_base_source,
                "faq_page_exists": technical.faq_page_exists,
                "faq_page_source": technical.faq_page_source,
                "key_info_in_html": technical.key_info_in_html,
                "key_info_in_html_source": technical.key_info_in_html_source,
                "content_structure_good": technical.content_structure_good,
                "content_structure_source": technical.content_structure_source,
                # 1.4 AI专属优化
                "llms_txt_exists": technical.llms_txt_exists,
                "llms_txt_source": technical.llms_txt_source,
                "robots_txt_allows_ai_bots": technical.robots_txt_allows_ai_bots,
                "robots_txt_allows_ai_bots_source": technical.robots_txt_allows_ai_bots_source,
                "eeat_signals": technical.eeat_signals,
                "eeat_signals_source": technical.eeat_signals_source,
                # 1.5 基础SEO
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


def report_to_markdown(report: Dict) -> str:
    """将JSON报告转换为Markdown格式"""
    md = []
    
    # 标题
    md.append(f"# GEO官网监测报告 - {report['metadata']['website']}")
    md.append(f"\n**品牌**: {report['metadata']['brand']}")
    md.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"\n**工具版本**: {report['metadata']['tool_version']}")
    md.append(f"\n**论文依据**: {report['metadata']['paper_info']['title']} ({report['metadata']['paper_info']['conference']})")
    
    # 技术审计
    md.append("\n## 1. 技术审计")
    tech = report['technical_audit']
    
    md.append("\n### 1.1 技术基础设施")
    md.append("\n| 检查项 | 结果 | 来源 |")
    md.append("|--------|------|------|")
    md.append(f"| HTTPS配置 | {'✅' if tech['https_ok'] else '❌'} | {tech['https_source']} |")
    md.append(f"| robots.txt | {'✅' if tech['robots_txt_exists'] else '❌'} | {tech['robots_txt_source']} |")
    md.append(f"| robots.txt允许AI | {'✅' if tech['robots_txt_allows_ai'] else '❌'} | {tech['robots_txt_ai_source']} |")
    md.append(f"| Sitemap | {'✅' if tech['sitemap_exists'] else '❌'} | {tech['sitemap_source']} |")
    md.append(f"| 服务器响应 | {'✅' if tech['server_response_ok'] else '❌'} ({tech.get('server_response_time', 'N/A')}秒) | {tech['server_response_source']} |")
    md.append(f"| 移动端适配 | {'✅' if tech['mobile_friendly'] else '❌'} | {tech['mobile_friendly_source']} |")
    
    md.append("\n### 1.2 结构化数据与Schema")
    md.append("\n| 检查项 | 结果 | 来源 |")
    md.append("|--------|------|------|")
    md.append(f"| Schema.org标记 | {'✅' if tech['schema_exists'] else '❌'} | {tech['schema_source']} |")
    md.append(f"| Organization Schema | {'✅' if tech['schema_organization'] else '❌'} | {tech['schema_organization_source']} |")
    md.append(f"| 核心Schema类型 | {', '.join(tech['schema_core_types']) if tech['schema_core_types'] else '无'} | {tech['schema_core_types_source']} |")
    md.append(f"| Schema关键属性 | {'✅' if tech['schema_key_properties'] else '❌'} | {tech['schema_key_properties_source']} |")
    md.append(f"| Schema完整性 | {'✅' if tech['schema_complete'] else '❌'} | {tech['schema_complete_source']} |")
    
    md.append("\n### 1.3 内容结构")
    md.append("\n| 检查项 | 结果 | 来源 |")
    md.append("|--------|------|------|")
    md.append(f"| 首页完整性 | {'✅' if tech['homepage_complete'] else '❌'} | {tech['homepage_complete_source']} |")
    md.append(f"| 关于我们页面 | {'✅' if tech['about_page_exists'] else '❌'} | {tech['about_page_source']} |")
    md.append(f"| 服务中心页面 | {'✅' if tech['service_pages_exist'] else '❌'} | {tech['service_pages_source']} |")
    md.append(f"| 案例中心页面 | {'✅' if tech['case_pages_exist'] else '❌'} | {tech['case_pages_source']} |")
    md.append(f"| 知识库/博客 | {'✅' if tech['knowledge_base_exists'] else '❌'} | {tech['knowledge_base_source']} |")
    md.append(f"| FAQ页面 | {'✅' if tech['faq_page_exists'] else '❌'} | {tech['faq_page_source']} |")
    md.append(f"| 关键信息在HTML | {'✅' if tech['key_info_in_html'] else '❌'} | {tech['key_info_in_html_source']} |")
    md.append(f"| 内容结构良好 | {'✅' if tech['content_structure_good'] else '❌'} | {tech['content_structure_source']} |")
    
    md.append("\n### 1.4 AI专属优化")
    md.append("\n| 检查项 | 结果 | 来源 |")
    md.append("|--------|------|------|")
    md.append(f"| llms.txt | {'✅' if tech['llms_txt_exists'] else '❌'} | {tech['llms_txt_source']} |")
    md.append(f"| robots.txt允许AI爬虫 | {'✅' if tech['robots_txt_allows_ai_bots'] else '❌'} | {tech['robots_txt_allows_ai_bots_source']} |")
    md.append(f"| E-E-A-T信号 | {'✅' if tech['eeat_signals'] else '❌'} | {tech['eeat_signals_source']} |")
    
    md.append("\n### 1.5 基础SEO")
    md.append("\n| 检查项 | 结果 | 来源 |")
    md.append("|--------|------|------|")
    md.append(f"| 登录墙 | {'⚠️' if tech['login_wall'] else '✅'} | {tech['login_wall_source']} |")
    md.append(f"| 付费墙 | {'⚠️' if tech['paywall'] else '✅'} | {tech['paywall_source']} |")
    md.append(f"| 核心内容在HTML | {'✅' if tech['core_content_in_html'] else '❌'} | {tech['core_content_source']} |")
    md.append(f"| 标题标签 | {'✅' if tech['title_tag'] else '❌'} | {tech['title_source']} |")
    md.append(f"| Meta Keywords | {'✅' if tech['meta_keywords'] else '❌'} | {tech['meta_keywords_source']} |")
    md.append(f"| Meta Description | {'✅' if tech['meta_description'] else '❌'} | {tech['meta_description_source']} |")
    md.append(f"| OpenGraph | {'✅' if tech['opengraph'] else '❌'} | {tech['opengraph_source']} |")
    
    # 内容友好度评估
    md.append("\n## 2. 内容友好度评估")
    md.append(f"\n*方法依据: {report['content_evaluation']['methods_basis']}*")
    md.append("\n| 排名 | 优化方法 | 得分/10 | 来源 |")
    md.append("|------|---------|---------|------|")
    content = report['content_evaluation']
    # 按论文排名顺序
    method_order = ['quotation_addition', 'statistics_addition', 'cite_sources', 'fluency_optimization', 
                     'authoritative_tone', 'easy_to_understand', 'unique_words', 'technical_terms']
    for idx, method_id in enumerate(method_order, 1):
        method_data = content[method_id]
        method_info = method_data['method_info']
        score = method_data.get('score', method_data.get('risk_score', 'N/A'))
        md.append(f"| {idx} | {method_info['name_cn']} ({method_info['name_en']}) | {score} | {method_data['source']} |")
    # 关键词堆砌单独处理
    keyword_method = content['keyword_stuffing']
    keyword_info = keyword_method['method_info']
    md.append(f"| 9 | {keyword_info['name_cn']} ({keyword_info['name_en']}) | {keyword_method['risk_score']} | {keyword_method['source']} |")
    
    # 引用测试
    if 'citation_tests' in report and report['citation_tests']:
        md.append("\n## 3. 引用测试")
        # 引用总结
        if 'citation_summary' in report:
            summary = report['citation_summary']
            md.append(f"\n**总测试数**: {summary['total_tests']}")
            md.append(f"\n**引用数**: {summary['cited_count']}")
            md.append(f"\n**提及数**: {summary['mentioned_count']}")
            md.append(f"\n**引用率**: {summary['citation_rate']}")
        # 详细测试结果
        md.append("\n### 3.1 详细测试结果")
        md.append("\n| 序号 | 场景词 | 引用 | 提及 | 位置 | 引用URLs |")
        md.append("|------|--------|------|------|------|---------|")
        for idx, test in enumerate(report['citation_tests'], 1):
            cited = '✅' if test['cited'] else '❌'
            mentioned = '✅' if test['mentioned'] else '❌'
            position = test['position'] if test['position'] else 'N/A'
            cited_urls = ', '.join(test['cited_urls']) if test.get('cited_urls') else 'N/A'
            md.append(f"| {idx} | {test['scenario']} | {cited} | {mentioned} | {position} | {cited_urls} |")
        # 模型回答
        md.append("\n### 3.2 模型回答详情")
        for idx, test in enumerate(report['citation_tests'], 1):
            md.append(f"\n#### {idx}. {test['scenario']}")
            model_answer = test.get('model_answer', '未接入真实模型API，仅占位')
            md.append(f"\n```\n{model_answer}\n```")
            md.append(f"\n*引用来源: {test['cited_source']}*")
            md.append(f"\n*提及来源: {test['mentioned_source']}*")
            md.append(f"\n*位置来源: {test['position_source']}*")
    
    # 优化建议
    md.append("\n## 4. 优化建议")
    if 'recommendations' in report and report['recommendations']:
        for rec in report['recommendations']:
            md.append(f"\n### 4.{rec['priority']}. {rec.get('method', rec.get('method_cn', '优化建议'))}")
            if 'method_source' in rec:
                md.append(f"\n*方法依据: {rec['method_source']}*")
            md.append(f"\n**建议**: {rec['description']}")
            md.append(f"\n**建议来源**: {rec['description_source']}")
            md.append(f"\n**证据来源**: {rec['evidence_source']}")
    else:
        md.append("\n暂无优化建议。")
    
    # 附录
    md.append("\n## 5. 附录")
    md.append(f"\n- 论文链接: {report['metadata']['paper_info']['url']}")
    md.append(f"\n- arXiv ID: {report['metadata']['paper_info']['arxiv']}")
    
    return '\n'.join(md)


def create_feishu_doc(title: str, markdown_content: str, use_bot: bool = True) -> Optional[Dict]:
    """使用 lark-cli 创建飞书文档并写入内容
    
    Args:
        title: 文档标题
        markdown_content: Markdown 内容
        use_bot: 是否使用 bot 身份（False 则使用 user 身份）
    
    Returns:
        创建成功返回文档信息，失败返回 None
    """
    try:
        # 1. 确保身份模式正确
        if use_bot:
            subprocess.run(["lark-cli", "config", "strict-mode", "bot"], 
                          capture_output=True, text=True)
        else:
            subprocess.run(["lark-cli", "config", "strict-mode", "user"], 
                          capture_output=True, text=True)
        
        # 2. 创建临时文件保存 Markdown 内容
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', encoding='utf-8', delete=False) as f:
            f.write(markdown_content)
            temp_path = f.name
        
        # 3. 创建文档
        create_cmd = [
            "lark-cli", "docs", "+create",
            "--api-version", "v2",
            "--title", title,
            "--content", markdown_content,
            "--doc-format", "markdown"
        ]
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        
        # 4. 解析创建结果（lark-cli docs +create 不支持 --json，输出为纯文本）
        # 查找类似 "doc_id: PramdoD4yozXkwxr1tJcI56snve" 或 "url: https://..."
        output = result.stdout + result.stderr
        
        doc_id = None
        doc_url = None
        
        # 尝试从输出中提取
        for line in output.split('\n'):
            line = line.strip()
            if 'doc_id:' in line or 'document_id:' in line:
                doc_id = line.split(':')[-1].strip()
            elif 'url:' in line and 'feishu.cn' in line:
                doc_url = line.split('url:')[-1].strip()
        
        # 如果没找到，尝试用 API 方式
        if not doc_id or not doc_url:
            # 尝试用 API 方式创建
            api_payload = {"title": title}
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8', delete=False) as f:
                json.dump(api_payload, f, ensure_ascii=False)
                api_payload_path = f.name
            
            api_cmd = [
                "lark-cli", "api", "POST",
                "/open-apis/docx/v1/documents",
                "--data", f"@{api_payload_path}"
            ]
            api_result = subprocess.run(api_cmd, capture_output=True, text=True)
            
            try:
                api_response = json.loads(api_result.stdout)
                if api_response.get('code') == 0:
                    doc_id = api_response['data']['document']['document_id']
                    doc_url = api_response['data']['document']['url']
            except:
                pass
        
        if doc_id:
            return {
                "document_id": doc_id,
                "url": doc_url,
                "title": title,
                "temp_file": temp_path
            }
        else:
            print(f"⚠️ 文档创建可能失败，输出: {output}")
            return None
            
    except Exception as e:
        print(f"❌ 创建飞书文档失败: {e}")
        return None


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
    """从模型回答中提取URL。"""
    if not text:
        return []
    urls = re.findall(r"https?://[^\s\)\]\}，。；;、]+", text)
    return list(dict.fromkeys(urls))


def load_agent_citation_results(json_path: str, brand_name: str, target_url: str, limit: int = 5) -> List[CitationTestResult]:
    """加载Hermes Agent真实引用测试结果。

    支持JSON数组，或包含 citation_tests/results 键的JSON对象。每条至少应包含
    scenario 和 model_answer；若未提供 cited/mentioned/position/cited_urls，脚本会按
    目标域名URL与品牌名进行保守判断。
    """
    limit = min(max(int(limit), 1), 5)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        rows = payload.get("citation_tests") or payload.get("results") or []
    else:
        rows = payload
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
        cited_urls = row.get("cited_urls") or _extract_urls(answer)
        cited_urls = [str(u) for u in cited_urls]
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
    parser.add_argument("--feishu", action="store_true", help="完成后创建飞书文档（可选）")
    parser.add_argument("--feishu-user", action="store_true", help="使用用户身份创建飞书文档（默认使用bot身份）")

    args = parser.parse_args()

    monitor = GEOWebsiteMonitor(args.url, args.brand)

    if args.mode == "audit":
        # 仅技术审计
        result = monitor.run_technical_audit()
        report = {
            "metadata": {
                "website": args.url,
                "brand": monitor.brand_name,
                "paper_info": GEO_PAPER_INFO,
                "tool_version": "3.0 (GEO白皮书优化版)"
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
                "paper_info": GEO_PAPER_INFO,
                "tool_version": "3.0 (GEO白皮书优化版)"
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
    
    # 创建飞书文档
    if args.feishu:
        print("\n正在创建飞书文档...")
        title = f"GEO 官网监测报告 - {monitor.brand_name}"
        markdown_content = report_to_markdown(report)
        doc_info = create_feishu_doc(title, markdown_content, use_bot=not args.feishu_user)
        if doc_info:
            print(f"\n✅ 飞书文档创建成功!")
            print(f"📄 文档标题: {doc_info['title']}")
            if doc_info['url']:
                print(f"🔗 文档链接: {doc_info['url']}")
            if doc_info['document_id']:
                print(f"📋 文档ID: {doc_info['document_id']}")
        else:
            print("\n⚠️ 飞书文档创建失败，请检查 lark-cli 配置。")


if __name__ == "__main__":
    main()