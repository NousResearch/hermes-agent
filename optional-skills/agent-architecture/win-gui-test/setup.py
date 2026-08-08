from setuptools import setup, find_packages

setup(
    name="win-gui-test-skill",
    version="1.0.0",
    description="Windows GUI automation test & visual analysis tool",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Hermes Agent",
    url="https://github.com/<your-username>/win-gui-test-skill",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pywinauto>=0.6.8,<1.0",
        "opencv-python>=4.5,<5.0",
        "pillow>=9.0,<12.0",
        "mss>=6.0,<10.0",
        "numpy>=1.20,<2.0",
        "pyyaml>=6.0,<7.0",
    ],
    entry_points={
        "console_scripts": [
            "win-gui-test=scripts.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
