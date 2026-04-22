package com.tencent.qidian.tmc.demo.springboot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Minimal Spring Boot demo application for webhook reception.
 */
@SpringBootApplication
public class WebhookDemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(WebhookDemoApplication.class, args);
    }
}
